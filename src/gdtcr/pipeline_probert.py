import argparse
import os
import re
import pickle
import warnings
from typing import Union

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tape import ProteinBertModel, TAPETokenizer
import torch_optimizer as optim

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from scipy.sparse import SparseEfficiencyWarning

try:
    import umap
except ImportError:
    umap = None

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


# ============================================================
# Dataset
# ============================================================
class CDR3Dataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_col: str = "sequence",
        tokenizer: Union[str, TAPETokenizer] = "unirep",
        max_length: int = 45,
    ):
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences, self.raw_data, self.columns = self.load_sequences(
            file_path=file_path,
            seq_col=seq_col,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        raw_info = self.raw_data[idx]

        token_ids = self.tokenizer.encode(sequence)

        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]

        input_mask = np.ones_like(token_ids)

        if len(token_ids) < self.max_length:
            pad_len = self.max_length - len(token_ids)
            token_ids = np.pad(token_ids, (0, pad_len), mode="constant", constant_values=0)
            input_mask = np.pad(input_mask, (0, pad_len), mode="constant", constant_values=0)

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = torch.tensor(input_mask, dtype=torch.long)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "raw_info": raw_info,
        }

    @staticmethod
    def load_sequences(file_path: str, seq_col: str):
        df = pd.read_csv(file_path)

        if seq_col not in df.columns:
            raise ValueError(
                f"Sequence column '{seq_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df[seq_col] = df[seq_col].astype(str)

        sequences = df[seq_col].tolist()
        raw_data = df.astype(str).values.tolist()
        columns = list(df.columns)

        return sequences, raw_data, columns


# ============================================================
# Model
# ============================================================
class CDR3BERT(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, input_mask=attention_mask)
        last_hidden_state = outputs[0]
        cls_embedding = last_hidden_state[:, 0, :]
        return last_hidden_state, cls_embedding


def build_model(proteinbert_path: str, device):
    bert_model = ProteinBertModel.from_pretrained(proteinbert_path).to(device)
    model = CDR3BERT(bert_model).to(device)
    return model


# ============================================================
# Training
# ============================================================
def train_model(model, data_loader, device, epochs, mask_probability=0.10, lr=1e-5):
    model.train()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.RAdam(model.parameters(), lr=lr)

    cls_embeddings_last_epoch = []
    sample_info_all = []

    for epoch in range(epochs):
        total_loss = 0.0

        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            raw_info = list(map(list, zip(*batch["raw_info"])))

            rand = torch.rand(input_ids.shape, device=device)
            mask = (rand < mask_probability) & (input_ids != 0)

            input_ids = input_ids.clone()
            labels = labels.clone()

            input_ids[mask] = 3
            labels[~mask] = -100

            logits, cls_embeddings = model(input_ids, attention_mask)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if epoch == epochs - 1:
                cls_embeddings_last_epoch.append(cls_embeddings.detach().cpu())
                sample_info_all.append(raw_info)

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    cls_embeddings = torch.cat(cls_embeddings_last_epoch, dim=0).numpy()
    sample_info = np.vstack(sample_info_all)

    return cls_embeddings, sample_info


def run_train(args):
    os.makedirs(args.output_path, exist_ok=True)

    model_path = os.path.join(args.output_path, "model.pth")
    embedding_path = os.path.join(args.output_path, "embeddings_train.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = TAPETokenizer(vocab=args.tokenizer)

    dataset = CDR3Dataset(
        file_path=args.data_path,
        seq_col=args.seq_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = build_model(args.proteinbert_path, device)

    cls_embeddings, sample_info = train_model(
        model=model,
        data_loader=data_loader,
        device=device,
        epochs=args.epochs,
        mask_probability=args.mask_probability,
        lr=args.lr,
    )

    torch.save(model.state_dict(), model_path)
    print(f"Saved model: {model_path}")

    with open(embedding_path, "wb") as f:
        pickle.dump(
            {
                "cls_embeddings": cls_embeddings,
                "sample_info": sample_info,
                "columns": dataset.columns,
            },
            f,
        )

    print(f"Saved training embeddings: {embedding_path}")


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(model, data_loader, device):
    model.eval()

    all_embeddings = []
    sample_info_all = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            raw_info = list(map(list, zip(*batch["raw_info"])))

            _, cls_embeddings = model(input_ids, attention_mask)

            all_embeddings.append(cls_embeddings.detach().cpu())
            sample_info_all.append(raw_info)

    cls_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    sample_info = np.vstack(sample_info_all)

    return cls_embeddings, sample_info


def run_eval(args):
    os.makedirs(args.output_path, exist_ok=True)

    embedding_path = os.path.join(args.output_path, "embeddings_eval.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = TAPETokenizer(vocab=args.tokenizer)

    dataset = CDR3Dataset(
        file_path=args.data_path,
        seq_col=args.seq_col,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(args.proteinbert_path, device)

    map_location = device if torch.cuda.is_available() else torch.device("cpu")
    state_dict = torch.load(args.model_path, map_location=map_location)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded model: {args.model_path}")

    cls_embeddings, sample_info = evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    with open(embedding_path, "wb") as f:
        pickle.dump(
            {
                "cls_embeddings": cls_embeddings,
                "sample_info": sample_info,
                "columns": dataset.columns,
            },
            f,
        )

    print(f"Saved evaluation embeddings: {embedding_path}")


# ============================================================
# Plot helpers
# ============================================================
def clean_gene_name(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x == "" or x.upper() == "NAN":
        return np.nan

    parts = x.split("|")
    parts = [p.strip().split("*")[0] for p in parts if p.strip() != ""]

    return "|".join(parts)


def clean_binding_name(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x in ["", "nan", "NaN", "None"]:
        return np.nan

    x = re.sub(r"Peptide-\d+", "Peptide", x)

    if x == "MR1-side":
        x = "MR1"

    return x


def load_embedding_pickle(input_pkl):
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    cls_embeddings = data["cls_embeddings"]
    sample_info = data["sample_info"]
    columns = data.get("columns", None)

    if columns is None:
        raise ValueError(
            "No column names found in pickle. "
            "Please regenerate embeddings using this pipeline."
        )

    return cls_embeddings, sample_info, columns


def get_col(sample_info, columns, col_name):
    if col_name not in columns:
        raise ValueError(
            f"Column '{col_name}' not found. Available columns: {columns}"
        )

    return sample_info[:, columns.index(col_name)]


def reduce_embeddings(
    cls_embeddings,
    method="umap",
    n_components=2,
    random_state=42,
    perplexity=30,
    n_neighbors=15,
):
    scaler = StandardScaler()
    x = scaler.fit_transform(cls_embeddings)

    method = method.lower()

    if method == "umap":
        if umap is None:
            raise ImportError("Install UMAP with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    if method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=200,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    if method == "pca":
        reducer = PCA(
            n_components=n_components,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    if method == "isomap":
        reducer = Isomap(
            n_neighbors=n_neighbors,
            n_components=n_components,
        )
        return reducer.fit_transform(x)

    if method == "lle":
        reducer = LLE(
            n_neighbors=n_neighbors,
            n_components=n_components,
            random_state=random_state,
        )
        return reducer.fit_transform(x)

    raise ValueError("method must be one of: umap, tsne, pca, isomap, lle")


def build_clean_plot_df(sample_info, columns, args):
    df = pd.DataFrame(sample_info, columns=columns)

    if args.binding_col in df.columns:
        df[args.binding_col] = df[args.binding_col].apply(clean_binding_name)

    if args.v_col in df.columns:
        df[args.v_col] = df[args.v_col].apply(clean_gene_name)

    if args.j_col in df.columns:
        df[args.j_col] = df[args.j_col].apply(clean_gene_name)

    return df


# ============================================================
# Plot functions
# ============================================================
def plot_embedding_from_df(
    embedding,
    df,
    output_dir,
    mode="binding",
    sequence_col="sequence",
    binding_col="binding",
    v_col="V",
    j_col="J",
    exclude_binding_pattern="OVA",
    show=False,
):
    os.makedirs(output_dir, exist_ok=True)

    if len(df) != embedding.shape[0]:
        raise ValueError(
            f"DataFrame rows ({len(df)}) != embedding rows ({embedding.shape[0]})"
        )

    df = df.copy()

    if binding_col in df.columns and exclude_binding_pattern is not None:
        keep_mask = ~df[binding_col].astype(str).str.contains(
            exclude_binding_pattern,
            na=False,
        )
        df = df.loc[keep_mask].copy()
        embedding = embedding[keep_mask.values]

    if mode == "binding":
        color_by = binding_col
    elif mode == "V":
        color_by = v_col
    elif mode == "J":
        color_by = j_col
    else:
        raise ValueError("mode must be one of: binding, V, J")

    if color_by not in df.columns:
        raise ValueError(f"Column '{color_by}' not found.")

    mask = (
        df[color_by].notna()
        & (df[color_by].astype(str).str.strip() != "")
        & (df[color_by].astype(str).str.upper() != "NAN")
    )

    emb_fg = embedding[mask.values]
    emb_bg = embedding[~mask.values]
    labels = df.loc[mask, color_by].astype(str).values

    if sequence_col in df.columns:
        sequences = df.loc[mask, sequence_col].values
    else:
        sequences = np.arange(mask.sum())

    if len(labels) == 0:
        print(f"No valid labels for mode: {mode}")
        return pd.DataFrame()

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    exact_color_map = {
        "Annexin A2": "#00BCE3",
        "BTN2A1": "#1F77B4",
        "BTNL3": "#C7E9C0",
        "CD1a": "#2CA02C",
        "CD1b": "#D62728",
        "CD1c": "#9467BD",
        "CD1d": "#8C564B",
        "EcIF1": "#E377C2",
        "HLA-A": "#E377C2",
        "HLA-DRA": "#7F7F7F",
        "MICA": "#17BECF",
        "MR1": "#BCBD22",
        "Phycoerythrin": "#F1B6DA",
        "Peptide": "#E6D083",
        "ULBP4": "#FF7F0E",
        "Pyrophosphate": "#FBB4AE",
        "Cy3_peptide": "#2CA02C",
    }

    exact_shape_map = {
        "Annexin A2": "v",
        "BTN2A1": "v",
        "BTNL3": "v",
        "CD1a": "s",
        "CD1b": "s",
        "CD1c": "s",
        "CD1d": "s",
        "EcIF1": "^",
        "HLA-A": "D",
        "HLA-DRA": "D",
        "MICA": "<",
        "MR1": ">",
        "Phycoerythrin": "X",
        "Peptide": "P",
        "ULBP4": "h",
        "Pyrophosphate": "o",
        "Cy3_peptide": "P",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    handles = []

    # background / non-labeled points are rasterized
    if len(emb_bg) > 0:
        ax.scatter(
            emb_bg[:, 0],
            emb_bg[:, 1],
            c="#999999",
            s=10,
            alpha=0.1,
            marker="o",
            rasterized=True,
        )

    if mode == "binding":
        for label_name in label_encoder.classes_:
            current_mask = labels == label_name
            pts = emb_fg[current_mask]

            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                c=exact_color_map.get(label_name, "#000000"),
                marker=exact_shape_map.get(label_name, "o"),
                label=label_name,
                s=100,
                alpha=0.9,
                edgecolor="black",
                linewidth=0.8,
                rasterized=False,
            )
            handles.append(sc)

    else:
        cmap = plt.get_cmap("tab20")

        for i, label_name in enumerate(label_encoder.classes_):
            current_mask = labels == label_name
            pts = emb_fg[current_mask]

            sc = ax.scatter(
                pts[:, 0],
                pts[:, 1],
                color=cmap(i % cmap.N),
                marker="o",
                label=label_name,
                s=10,
                alpha=0.85,
                edgecolor="none",
                rasterized=True,
            )
            handles.append(sc)

    if handles:
        leg = ax.legend(
            handles=handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.6,
        )

        for lh in leg.legend_handles:
            lh.set_alpha(1.0)

    ax.set_title(f"UMAP/t-SNE Colored by {mode}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"umap_{mode}.pdf")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)

    print(f"Saved: {out_pdf}")

    return pd.DataFrame(
        {
            sequence_col: sequences,
            color_by: labels,
            "Component 1": emb_fg[:, 0],
            "Component 2": emb_fg[:, 1],
        }
    )


def plot_all_modes_from_embedding(
    embedding,
    df,
    output_dir,
    modes=("binding", "V", "J"),
    **kwargs,
):
    results = {}

    for mode in modes:
        print(f"Plotting mode: {mode}")
        results[mode] = plot_embedding_from_df(
            embedding=embedding,
            df=df,
            output_dir=output_dir,
            mode=mode,
            **kwargs,
        )

    print(f"All plots saved to: {output_dir}")
    return results


def plot_distance_to_bulk(
    embedding_2d,
    df,
    output_pdf,
    threshold=7,
    group_col="group",
):
    if group_col not in df.columns:
        print(f"Skipping distance filter: column '{group_col}' not found.")
        return np.arange(len(df))

    group = df[group_col].astype(str).values

    bulk_idx = np.where(group == "bulk")[0]
    non_bulk_idx = np.where(group != "bulk")[0]

    if len(bulk_idx) == 0 or len(non_bulk_idx) == 0:
        print("Skipping distance filter: need both bulk and non-bulk samples.")
        return np.arange(len(df))

    bulk_emb = embedding_2d[bulk_idx]
    non_bulk_emb = embedding_2d[non_bulk_idx]

    distances = cdist(non_bulk_emb, bulk_emb, metric="euclidean")
    min_distances = np.min(distances, axis=1)

    plt.figure(figsize=(8, 6))
    plt.hist(min_distances, bins=30, edgecolor="black")
    plt.axvline(threshold, linestyle="--", label=f"threshold={threshold}")
    plt.title("Minimum distance of non-bulk samples to bulk samples")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_pdf, format="pdf")
    plt.close()

    keep_non_bulk_idx = non_bulk_idx[min_distances < threshold]
    keep_idx = np.concatenate([bulk_idx, keep_non_bulk_idx])

    print(f"Saved distance plot: {output_pdf}")
    print(f"Kept {len(keep_idx)} / {len(df)} samples after filtering.")

    return keep_idx


def run_plot(args):
    os.makedirs(args.output_path, exist_ok=True)

    cls_embeddings, sample_info, columns = load_embedding_pickle(args.input_pkl)

    df = build_clean_plot_df(sample_info, columns, args)

    embedding_2d = reduce_embeddings(
        cls_embeddings,
        method=args.method,
        random_state=args.random_state,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
    )

    raw_npy = os.path.join(args.output_path, f"{args.method}.npy")
    np.save(raw_npy, embedding_2d)
    print(f"Saved reduced embedding npy: {raw_npy}")

    if args.skip_filter:
        keep_idx = np.arange(len(df))
    else:
        keep_idx = plot_distance_to_bulk(
            embedding_2d=embedding_2d,
            df=df,
            output_pdf=os.path.join(args.output_path, f"{args.method}_distance_to_bulk.pdf"),
            threshold=args.threshold,
            group_col=args.group_col,
        )

    embedding_filtered = embedding_2d[keep_idx]
    df_filtered = df.iloc[keep_idx].copy()

    filtered_npy = os.path.join(args.output_path, f"{args.method}_filtered.npy")
    np.save(filtered_npy, embedding_filtered)
    print(f"Saved filtered reduced embedding npy: {filtered_npy}")

    reduced_csv = os.path.join(args.output_path, "reduced_coordinates.csv")
    df_out = df_filtered.copy()
    df_out["Component 1"] = embedding_filtered[:, 0]
    df_out["Component 2"] = embedding_filtered[:, 1]
    df_out.to_csv(reduced_csv, index=False)
    print(f"Saved reduced coordinates: {reduced_csv}")

    plot_all_modes_from_embedding(
        embedding=embedding_filtered,
        df=df_filtered,
        output_dir=args.output_path,
        modes=("binding", "V", "J"),
        sequence_col=args.seq_col,
        binding_col=args.binding_col,
        v_col=args.v_col,
        j_col=args.j_col,
        exclude_binding_pattern=args.exclude_binding_pattern,
        show=False,
    )


def run_plot_from_npy(args):
    os.makedirs(args.output_path, exist_ok=True)

    embedding = np.load(args.npy_file)
    df = pd.read_csv(args.labels_csv)

    if args.binding_col in df.columns:
        df[args.binding_col] = df[args.binding_col].apply(clean_binding_name)

    if args.v_col in df.columns:
        df[args.v_col] = df[args.v_col].apply(clean_gene_name)

    if args.j_col in df.columns:
        df[args.j_col] = df[args.j_col].apply(clean_gene_name)

    plot_all_modes_from_embedding(
        embedding=embedding,
        df=df,
        output_dir=args.output_path,
        modes=("binding", "V", "J"),
        sequence_col=args.seq_col,
        binding_col=args.binding_col,
        v_col=args.v_col,
        j_col=args.j_col,
        exclude_binding_pattern=args.exclude_binding_pattern,
        show=False,
    )


# ============================================================
# Run all
# ============================================================
def run_all(args):
    os.makedirs(args.output_path, exist_ok=True)

    train_dir = os.path.join(args.output_path, "train")
    eval_dir = os.path.join(args.output_path, "eval")
    plot_dir = os.path.join(args.output_path, "plot")

    train_args = argparse.Namespace(**vars(args))
    train_args.output_path = train_dir
    run_train(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.model_path = os.path.join(train_dir, "model.pth")
    eval_args.output_path = eval_dir
    run_eval(eval_args)

    plot_args = argparse.Namespace(**vars(args))
    plot_args.input_pkl = os.path.join(eval_dir, "embeddings_eval.pkl")
    plot_args.output_path = plot_dir
    run_plot(plot_args)


# ============================================================
# CLI
# ============================================================
def add_common_data_args(parser):
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--seq_col", type=str, default="sequence")
    parser.add_argument("--v_col", type=str, default="V")
    parser.add_argument("--j_col", type=str, default="J")
    parser.add_argument("--binding_col", type=str, default="binding")
    parser.add_argument("--group_col", type=str, default="group")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=45)
    parser.add_argument("--tokenizer", type=str, default="unirep")
    parser.add_argument("--proteinbert_path", type=str, default="../models/proteinbert")


def add_plot_args(parser):
    parser.add_argument(
        "--method",
        type=str,
        default="umap",
        choices=["umap", "tsne", "pca", "isomap", "lle"],
    )
    parser.add_argument("--threshold", type=float, default=7)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--skip_filter", action="store_true")
    parser.add_argument("--exclude_binding_pattern", type=str, default="OVA")


def main():
    parser = argparse.ArgumentParser(
        description="ProteinBERT CDR3 training, evaluation, and plotting pipeline."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    add_common_data_args(train_parser)
    train_parser.add_argument("--output_path", type=str, required=True)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--mask_probability", type=float, default=0.10)
    train_parser.add_argument("--lr", type=float, default=1e-5)
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("eval")
    add_common_data_args(eval_parser)
    eval_parser.add_argument("--model_path", type=str, required=True)
    eval_parser.add_argument("--output_path", type=str, required=True)
    eval_parser.set_defaults(func=run_eval)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("--input_pkl", type=str, required=True)
    plot_parser.add_argument("--output_path", type=str, required=True)
    plot_parser.add_argument("--seq_col", type=str, default="sequence")
    plot_parser.add_argument("--v_col", type=str, default="V")
    plot_parser.add_argument("--j_col", type=str, default="J")
    plot_parser.add_argument("--binding_col", type=str, default="binding")
    plot_parser.add_argument("--group_col", type=str, default="group")
    add_plot_args(plot_parser)
    plot_parser.set_defaults(func=run_plot)

    plot_npy_parser = subparsers.add_parser("plot-npy")
    plot_npy_parser.add_argument("--npy_file", type=str, required=True)
    plot_npy_parser.add_argument("--labels_csv", type=str, required=True)
    plot_npy_parser.add_argument("--output_path", type=str, required=True)
    plot_npy_parser.add_argument("--seq_col", type=str, default="sequence")
    plot_npy_parser.add_argument("--v_col", type=str, default="V")
    plot_npy_parser.add_argument("--j_col", type=str, default="J")
    plot_npy_parser.add_argument("--binding_col", type=str, default="binding")
    plot_npy_parser.add_argument("--group_col", type=str, default="group")
    plot_npy_parser.add_argument("--exclude_binding_pattern", type=str, default="OVA")
    plot_npy_parser.set_defaults(func=run_plot_from_npy)

    all_parser = subparsers.add_parser("all")
    add_common_data_args(all_parser)
    all_parser.add_argument("--output_path", type=str, required=True)
    all_parser.add_argument("--epochs", type=int, default=5)
    all_parser.add_argument("--mask_probability", type=float, default=0.10)
    all_parser.add_argument("--lr", type=float, default=1e-5)
    add_plot_args(all_parser)
    all_parser.set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()