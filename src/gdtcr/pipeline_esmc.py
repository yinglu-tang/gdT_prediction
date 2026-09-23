"""
esmc_pipeline.py
================
A clean, modular pipeline for ESMC MLM fine-tuning, CLS embedding extraction,
and dimensionality-reduction plotting.

Output layout is intentionally simple:

    target_dir/
        model.pt
        run_summary.json
        embeddings/
            batch_00000.pt
            batch_00001.pt
            ...
        plots/
            umap.pdf
            umap.npy
            tsne.pdf
            tsne.npy
            ...

The summary file records timestamp, random seed, device, parameters, and output paths.
Plots are always saved inside target_dir/plots, even if you call only pipe.plot().
Embeddings are always saved inside target_dir/embeddings, even if you call only pipe.embed().

Quick start:
------------
    from esmc_pipeline import ESMCPipeline

    pipe = ESMCPipeline(
        data_csv="path/to/sequences.csv",
        target_dir="path/to/output/",
        seed=42,
    )
    pipe.run()

Call stages separately:
-----------------------
    pipe.train()
    df_emb = pipe.embed()
    pipe.plot(methods=["umap", "tsne"])

Plot only from saved embeddings:
--------------------------------
    pipe = ESMCPipeline(
        data_csv="path/to/sequences.csv",   # needed for labels unless df_label is passed
        target_dir="path/to/output/",
        seed=42,
    )
    pipe.plot()
"""

# ── standard library ─────────────────────────────────────────────────────────
import glob
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

# ── third-party ───────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from adjustText import adjust_text
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from umap import UMAP

# ── ESM ───────────────────────────────────────────────────────────────────────
from esm.models.esmc import ESMC
from esm.utils.constants.models import ESMC_600M


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class _MaskedSequenceDataset(Dataset):
    """BERT-style masked-language-model dataset for amino acid sequences."""

    def __init__(self, sequences: List[str], tokenizer, max_len: int, mask_prob: float):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        tokens = self.tokenizer(
            seq,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Slightly higher mask probability for very short sequences.
        mask_prob = min(self.mask_prob, max(0.05, 2.0 / input_ids.size(0)))
        prob_matrix = torch.full(input_ids.shape, mask_prob)

        # Never mask special tokens.
        special_ids = [
            t for t in [
                self.tokenizer.pad_token_id,
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
            ] if t is not None
        ]
        if special_ids:
            prob_matrix.masked_fill_(
                torch.isin(input_ids, torch.tensor(special_ids)), 0.0
            )

        masked = torch.bernoulli(prob_matrix).bool()
        labels[~masked] = -100

        corrupted = input_ids.clone()

        # 80% -> [MASK]
        replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked
        if self.tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no [MASK] token.")
        corrupted[replace_mask] = self.tokenizer.mask_token_id

        # 10% -> random token
        replace_rand = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked
            & ~replace_mask
        )
        corrupted[replace_rand] = torch.randint(
            0, self.tokenizer.vocab_size, input_ids.shape
        )[replace_rand]

        # Remaining 10% -> keep original, but still predict it.
        return corrupted, labels


def _make_collate_fn(pad_token_id: int):
    """Return a collate function that pads each batch to its own max length."""

    def collate(batch):
        input_ids_list, labels_list = zip(*batch)
        max_len = max(len(x) for x in input_ids_list)

        def pad(tensors, val):
            return torch.stack([
                torch.cat([x, torch.full((max_len - len(x),), val, dtype=torch.long)])
                for x in tensors
            ])

        return pad(input_ids_list, pad_token_id), pad(labels_list, -100)

    return collate


def _reduce_dimensions(X: np.ndarray, method: str, seed: int) -> np.ndarray:
    """Project X, shape N x D, to 2-D using PCA, t-SNE, or UMAP."""
    method = method.lower()
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    if method == "tsne":
        # Keep perplexity valid for small datasets.
        perplexity = min(30, max(2, (len(X) - 1) // 3))
        return TSNE(
            n_components=2,
            random_state=seed,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X)
    if method == "umap":
        n_neighbors = min(15, max(2, len(X) - 1))
        return UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors).fit_transform(X)
    raise ValueError(f"Unsupported method: {method!r}. Choose pca, tsne, or umap.")


def _prefix_before_star(x) -> Optional[str]:
    """Convert 'TRDV1*01' to 'TRDV1'; handles NaN."""
    if pd.isna(x):
        return None
    x = str(x).strip()
    return x.split("*", 1)[0] if "*" in x else x


def _now_stamp() -> str:
    """Compact timestamp used inside the summary file, not in file names."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json_default(obj):
    """Make common scientific/Python objects JSON serializable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, torch.device):
        return str(obj)
    return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  ESMCPipeline — public API
# ══════════════════════════════════════════════════════════════════════════════

class ESMCPipeline:
    """
    End-to-end pipeline: MLM fine-tuning -> CLS embedding extraction -> plotting.

    Parameters
    ----------
    data_csv
        Path to CSV with a 'sequence' column, and optionally 'group', 'label',
        'binding', 'V', and 'J'.
    target_dir
        One root folder for all outputs.
    seed
        Integer random seed. If None, a random seed is generated and recorded.
    batch_size
        Training mini-batch size.
    accum_steps
        Gradient accumulation steps.
    lr
        Learning rate.
    max_len
        Maximum tokenized length.
    mask_prob
        Base MLM mask probability.
    epochs
        Number of training epochs.
    """

    def __init__(
        self,
        data_csv: str = "",
        target_dir: str = "esmc_output",
        seed: Optional[int] = None,
        batch_size: int = 64,
        accum_steps: int = 2,
        lr: float = 1e-4,
        max_len: int = 40,
        mask_prob: float = 0.15,
        epochs: int = 5,
    ):
        self.data_csv = data_csv
        self.target_dir = os.path.abspath(target_dir)

        # Simple, stable output paths.
        self.save_path = os.path.join(self.target_dir, "model.pt")
        self.embed_dir = os.path.join(self.target_dir, "embeddings")
        self.plot_dir = os.path.join(self.target_dir, "plots")
        self.summary_path = os.path.join(self.target_dir, "run_summary.json")

        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.lr = lr
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.epochs = epochs

        # Fix seed.
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.target_dir, exist_ok=True)
        os.makedirs(self.embed_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # Lazy-loaded attributes.
        self._model = None
        self._tokenizer = None
        self._df = None

        self._write_summary(
            stage="init",
            details={
                "message": "Pipeline initialized.",
                "data_csv": self.data_csv,
                "target_dir": self.target_dir,
                "model_file": self.save_path,
                "embedding_dir": self.embed_dir,
                "plot_dir": self.plot_dir,
                "batch_size": self.batch_size,
                "accum_steps": self.accum_steps,
                "lr": self.lr,
                "max_len": self.max_len,
                "mask_prob": self.mask_prob,
                "epochs": self.epochs,
            },
        )
        print(f"[ESMCPipeline] seed={self.seed} | device={self.device}")
        print(f"[ESMCPipeline] target_dir={self.target_dir}")

    # ── summary helper ───────────────────────────────────────────────────────

    def _write_summary(self, stage: str, details: Optional[Dict] = None) -> None:
        """
        Update target_dir/run_summary.json.

        This file is updated whether you call train(), embed(), plot(), or run().
        Timestamp and random seed are recorded here instead of being placed in
        folder/file names.
        """
        summary = {}
        if os.path.isfile(self.summary_path):
            try:
                with open(self.summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                summary = {}

        summary.update({
            "last_updated": _now_stamp(),
            "seed": self.seed,
            "device": str(self.device),
            "target_dir": self.target_dir,
            "data_csv": self.data_csv,
            "model_file": self.save_path,
            "embedding_dir": self.embed_dir,
            "plot_dir": self.plot_dir,
        })

        history = summary.get("history", [])
        history.append({
            "timestamp": _now_stamp(),
            "stage": stage,
            "details": details or {},
        })
        summary["history"] = history

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default)

    # ── model access ─────────────────────────────────────────────────────────

    @property
    def model(self) -> ESMC:
        if self._model is None:
            print("[ESMCPipeline] Loading ESMC-600M ...")
            self._model = ESMC.from_pretrained(ESMC_600M).to(self.device)
            self._tokenizer = self._model.tokenizer
        return self._model

    @property
    def tokenizer(self):
        _ = self.model
        return self._tokenizer

    def load_weights(self) -> None:
        """Load fine-tuned weights from target_dir/model.pt into the model."""
        state = torch.load(
            self.save_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state)
        print(f"[ESMCPipeline] Weights loaded from {self.save_path}")

    # ── data helpers ─────────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        if self._df is None:
            if not self.data_csv:
                raise ValueError("data_csv is required for training, embedding, or default plotting labels.")
            self._df = (
                pd.read_csv(self.data_csv, low_memory=False)
                .dropna(subset=["sequence"])
                .assign(sequence=lambda d: d["sequence"].astype(str).str.strip())
            )
            print(f"[ESMCPipeline] Loaded {len(self._df)} rows from {self.data_csv}")
        return self._df

    # ══════════════════════════════════════════════════════════════════════════
    #  1. TRAIN
    # ══════════════════════════════════════════════════════════════════════════

    def train(self) -> dict:
        """
        Fine-tune ESMC with a masked-language-model objective.

        Returns
        -------
        dict
            {'train_losses': [...], 'eval_losses': [...]}.
        """
        df = self._load_csv()

        if "group" in df.columns:
            seqs = df.loc[df["group"] == "bulk", "sequence"].tolist()
            print(f"[train] Using {len(seqs)} bulk sequences.")
        else:
            seqs = df["sequence"].tolist()
            print(f"[train] Using all {len(seqs)} sequences.")

        if len(seqs) < 2:
            raise ValueError("Need at least two sequences for train/eval split.")

        train_seqs, eval_seqs = train_test_split(
            seqs,
            test_size=0.2,
            random_state=self.seed,
        )

        collate = _make_collate_fn(self.tokenizer.pad_token_id)
        g = torch.Generator().manual_seed(self.seed)

        train_loader = DataLoader(
            _MaskedSequenceDataset(train_seqs, self.tokenizer, self.max_len, self.mask_prob),
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
            collate_fn=collate,
        )
        eval_loader = DataLoader(
            _MaskedSequenceDataset(eval_seqs, self.tokenizer, self.max_len, self.mask_prob),
            batch_size=self.batch_size,
            shuffle=False,
            generator=g,
            collate_fn=collate,
        )
        print(f"[train] {len(train_loader)} train batches, {len(eval_loader)} eval batches.")

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        train_losses, eval_losses = [], []
        t0 = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_train = 0.0
            optimizer.zero_grad()

            for step, (tokens, labels) in enumerate(train_loader, 1):
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(sequence_tokens=tokens).sequence_logits
                loss = criterion(logits.permute(0, 2, 1), labels) / self.accum_steps
                loss.backward()
                total_train += loss.item() * self.accum_steps

                if step % self.accum_steps == 0 or step == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train = total_train / max(1, len(train_loader))

            self.model.eval()
            total_eval = 0.0
            with torch.no_grad():
                for tokens, labels in eval_loader:
                    tokens = tokens.to(self.device)
                    labels = labels.to(self.device)
                    logits = self.model(sequence_tokens=tokens).sequence_logits
                    total_eval += criterion(logits.permute(0, 2, 1), labels).item()

            avg_eval = total_eval / max(1, len(eval_loader))
            train_losses.append(avg_train)
            eval_losses.append(avg_eval)
            print(f"Epoch {epoch:02d} | Train Loss: {avg_train:.4f} | Eval Loss: {avg_eval:.4f}")

        elapsed_min = (time.time() - t0) / 60
        torch.save(self.model.state_dict(), self.save_path)
        print(f"[train] Completed in {elapsed_min:.2f} min.")
        print(f"[train] Weights saved to {self.save_path}")

        history = {"train_losses": train_losses, "eval_losses": eval_losses}
        self._write_summary(
            stage="train",
            details={
                "elapsed_min": elapsed_min,
                "n_train_sequences": len(train_seqs),
                "n_eval_sequences": len(eval_seqs),
                "train_losses": train_losses,
                "eval_losses": eval_losses,
                "model_file": self.save_path,
            },
        )
        return history

    # ══════════════════════════════════════════════════════════════════════════
    #  2. EMBED — last-layer CLS token only, original sequence order
    # ══════════════════════════════════════════════════════════════════════════

    def embed(
        self,
        sequences: Optional[List[str]] = None,
        batch_size: int = 64,
        save_every: int = 5000,
        clear_existing: bool = True,
    ) -> pd.DataFrame:
        """
        Extract last-layer CLS embeddings and save batch files under target_dir/embeddings.

        Parameters
        ----------
        sequences
            Amino-acid strings. If None, reads the sequence column from data_csv.
        batch_size
            Max sequences per forward pass within each same-length bucket.
        save_every
            Save one checkpoint every N successfully embedded sequences.
        clear_existing
            If True, remove old batch_*.pt files before embedding to avoid mixing runs.

        Returns
        -------
        pd.DataFrame
            Columns: ['sequence', 'cls_last'].
        """
        if sequences is None:
            sequences = self._load_csv()["sequence"].tolist()

        sequences = [str(s).strip() for s in sequences if pd.notna(s) and str(s).strip()]
        if len(sequences) == 0:
            print("[embed] No sequences found.")
            self._write_summary(stage="embed", details={"n_sequences": 0})
            return pd.DataFrame(columns=["sequence", "cls_last"])

        os.makedirs(self.embed_dir, exist_ok=True)
        if clear_existing:
            for old_file in glob.glob(os.path.join(self.embed_dir, "batch_*.pt")):
                os.remove(old_file)

        if os.path.isfile(self.save_path):
            self.load_weights()
        else:
            print(f"[embed] No fine-tuned model found at {self.save_path}; using current/pretrained model.")

        self.model.eval()

        # Group sequences by length, keeping original index.
        length_groups = defaultdict(list)
        for orig_idx, seq in enumerate(sequences):
            length_groups[len(seq)].append((orig_idx, seq))

        unique_lengths = sorted(length_groups.keys())
        print(
            f"[embed] {len(sequences)} sequences | "
            f"{len(unique_lengths)} unique lengths "
            f"(min={unique_lengths[0]}, max={unique_lengths[-1]}) | "
            f"batch_size={batch_size}"
        )

        batch_idx = 0
        processed = 0
        failed = 0
        checkpoint_buffer = []
        saved_files = []
        t0 = time.time()

        with torch.no_grad():
            for length in unique_lengths:
                items = length_groups[length]

                for i in range(0, len(items), batch_size):
                    chunk = items[i:i + batch_size]
                    orig_indices, batch_seqs = zip(*chunk)

                    try:
                        tokens = self.tokenizer(
                            list(batch_seqs),
                            add_special_tokens=True,
                            padding=False,      # same length in this bucket
                            truncation=True,
                            max_length=self.max_len,
                            return_tensors="pt",
                        )
                        input_ids = tokens["input_ids"].to(self.device)

                        out = self.model(sequence_tokens=input_ids)
                        hidden = out.hidden_states[-1]

                        # Expected shape for this call is (B, L, D), so CLS is [:, 0, :].
                        # If a future ESMC version returns (L, B, D), handle it safely.
                        if hidden.ndim != 3:
                            raise RuntimeError(f"Expected 3-D hidden state, got shape {tuple(hidden.shape)}")
                        if hidden.shape[0] == len(batch_seqs):
                            cls_batch = hidden[:, 0, :].float().cpu()
                        elif hidden.shape[1] == len(batch_seqs):
                            cls_batch = hidden[0, :, :].float().cpu()
                        else:
                            raise RuntimeError(
                                "Cannot infer batch axis from hidden state shape "
                                f"{tuple(hidden.shape)} for batch size {len(batch_seqs)}"
                            )

                        for orig_idx, seq, vec in zip(orig_indices, batch_seqs, cls_batch):
                            checkpoint_buffer.append({
                                "orig_idx": int(orig_idx),
                                "sequence": seq,
                                "cls_last": vec,
                            })

                        del tokens, input_ids, out, hidden, cls_batch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        failed += len(chunk)
                        print(f"[embed] WARNING: skipping batch length={length}, size={len(chunk)}: {e}")

                    processed += len(chunk)

                    if len(checkpoint_buffer) >= save_every or processed >= len(sequences):
                        if checkpoint_buffer:
                            batch_file = os.path.join(self.embed_dir, f"batch_{batch_idx:05d}.pt")
                            torch.save(checkpoint_buffer, batch_file)
                            saved_files.append(batch_file)
                            print(
                                f"[embed] {processed}/{len(sequences)} processed | "
                                f"saved {os.path.basename(batch_file)}"
                            )
                            batch_idx += 1
                            checkpoint_buffer = []

        elapsed_min = (time.time() - t0) / 60
        print("[embed] Done embedding. Reloading batches to return DataFrame...")
        df_emb = self.load_embeddings()

        self._write_summary(
            stage="embed",
            details={
                "elapsed_min": elapsed_min,
                "n_input_sequences": len(sequences),
                "n_embedded_sequences": len(df_emb),
                "n_failed_sequences": failed,
                "embedding_dir": self.embed_dir,
                "batch_files": saved_files,
                "clear_existing": clear_existing,
            },
        )
        return df_emb

    def load_embeddings(self) -> pd.DataFrame:
        """Reload and merge all target_dir/embeddings/batch_*.pt files."""
        files = sorted(glob.glob(os.path.join(self.embed_dir, "batch_*.pt")))
        if not files:
            raise FileNotFoundError(f"No batch_*.pt files found in {self.embed_dir}")

        records = []
        for f in files:
            part = torch.load(f, weights_only=False, map_location="cpu")
            records.extend(part)

        df = pd.DataFrame(records)
        if "orig_idx" in df.columns:
            df = (
                df.sort_values("orig_idx")
                .drop(columns=["orig_idx"])
                .reset_index(drop=True)
            )

        print(f"[load_embeddings] {len(df)} records from {len(files)} files.")
        return df

    # ══════════════════════════════════════════════════════════════════════════
    #  3. PLOT
    # ══════════════════════════════════════════════════════════════════════════

    def plot(
        self,
        df_embeddings: Optional[pd.DataFrame] = None,
        df_label: Optional[pd.DataFrame] = None,
        methods: List[str] = ("umap", "tsne"),
        modes: List[str] = ("original", "TRDV", "TRDJ"),
        start_layer: int = 0,
        num_layers: int = 1,
        base_dir: Optional[str] = None,
    ) -> dict:
        """
        Create dimensionality-reduction PDFs and save them inside target_dir/plots.

        Parameters
        ----------
        df_embeddings
            DataFrame with 'sequence' and 'cls_last'. If None, loads saved batches.
        df_label
            DataFrame with 'sequence' and label columns. If None, uses data_csv.
        methods
            Any subset of ['pca', 'tsne', 'umap'].
        modes
            Any subset of ['original', 'TRDV', 'TRDJ'].
        start_layer, num_layers
            Kept only for backward compatibility. This pipeline always plots cls_last.
        base_dir
            Optional override. If None, plots are saved to target_dir/plots.
            If provided, plots are saved directly to that folder, without nested
            timestamp/seed folders.

        Returns
        -------
        dict
            {method: {'folder': ..., 'combined_pdf': ..., 'projection': ...}}.
        """
        del start_layer, num_layers  # Backward-compatible arguments; not used.

        if df_embeddings is None:
            df_embeddings = self.load_embeddings()
        if df_label is None:
            df_label = self._load_csv()

        df_embeddings = df_embeddings.copy()
        df_embeddings["sequence"] = df_embeddings["sequence"].astype(str).str.strip()

        df_label = df_label.copy()
        df_label["sequence"] = df_label["sequence"].astype(str).str.strip()

        out_root = os.path.abspath(base_dir) if base_dir is not None else self.plot_dir
        os.makedirs(out_root, exist_ok=True)

        valid_rows = df_embeddings[df_embeddings["cls_last"].notna()].copy()
        if len(valid_rows) < 2:
            raise ValueError("Need at least two embeddings to make a 2-D plot.")

        def _to_1d(t):
            if not torch.is_tensor(t):
                t = torch.tensor(t)
            t = t.float().cpu()
            return t.squeeze() if t.ndim > 1 else t

        vecs = [_to_1d(t) for t in valid_rows["cls_last"].tolist()]
        shapes = {tuple(v.shape) for v in vecs}
        if len(shapes) > 1:
            raise RuntimeError(
                f"cls_last vectors have inconsistent shapes: {shapes}.\n"
                "Re-run pipe.embed(clear_existing=True) to regenerate embeddings."
            )
        X = torch.stack(vecs).numpy()

        results = {}
        for method in methods:
            method = method.lower()
            print(f"[plot] Running {method.upper()} on {len(X)} embeddings ...")
            proj = _reduce_dimensions(X, method, self.seed)

            # Simple names: no timestamp/seed in file names. Summary stores that metadata.
            proj_file = os.path.join(out_root, f"{method}.npy")
            pdf_path = os.path.join(out_root, f"{method}.pdf")

            np.save(proj_file, proj)

            with PdfPages(pdf_path) as pdf:
                for mode in modes:
                    fig, ax = plt.subplots(figsize=(7, 7))
                    self._plot_one(
                        proj=proj,
                        sequences=valid_rows["sequence"].tolist(),
                        df_label=df_label,
                        method=method,
                        mode=mode,
                        ax=ax,
                    )
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            print(f"[plot] {method.upper()} PDF saved -> {pdf_path}")
            results[method] = {
                "folder": out_root,
                "combined_pdf": pdf_path,
                "projection": proj_file,
            }

        self._write_summary(
            stage="plot",
            details={
                "n_embeddings": len(valid_rows),
                "methods": list(methods),
                "modes": list(modes),
                "plot_dir": out_root,
                "outputs": results,
            },
        )
        return results

    # ── low-level plotting helper ────────────────────────────────────────────

    def _plot_one(
        self,
        proj: np.ndarray,
        sequences: List[str],
        df_label: pd.DataFrame,
        method: str,
        mode: str,
        ax,
    ):
        """Draw one scatter panel onto ax."""
        mode = mode.lower()

        if mode == "original":
            label_col = "label" if "label" in df_label.columns else "binding"
            if label_col not in df_label.columns:
                raise ValueError("df_label needs 'label' or 'binding' for mode='original'.")
            seq2label = dict(zip(df_label["sequence"], df_label[label_col]))
            show_text = True
            legend_title = label_col
            highlight = {"Peptide", "ULBP4", "Vaccine"}

        elif mode == "trdv":
            if "V" not in df_label.columns:
                raise ValueError("df_label needs column 'V' for mode='TRDV'.")
            seq2label = dict(zip(df_label["sequence"], df_label["V"].apply(_prefix_before_star)))
            show_text = False
            legend_title = "TRDV"
            highlight = set()

        elif mode == "trdj":
            if "J" not in df_label.columns:
                raise ValueError("df_label needs column 'J' for mode='TRDJ'.")
            seq2label = dict(zip(df_label["sequence"], df_label["J"].apply(_prefix_before_star)))
            show_text = False
            legend_title = "TRDJ"
            highlight = set()

        else:
            raise ValueError(f"mode must be 'original', 'TRDV', or 'TRDJ'; got {mode!r}.")

        label_list = [seq2label.get(s) or "unlabeled" for s in sequences]
        label_list = [
            l if (l and not (isinstance(l, float) and np.isnan(l))) else "unlabeled"
            for l in label_list
        ]

        unique_labels = sorted(set(label_list) - {"unlabeled"}) + ["unlabeled"]
        palette = sns.color_palette("hls", max(1, len(unique_labels) - 1))
        color_map = {
            lbl: palette[i]
            for i, lbl in enumerate(unique_labels)
            if lbl != "unlabeled"
        }
        color_map["unlabeled"] = "gray"

        labeled_mask = np.array([l != "unlabeled" for l in label_list])
        labeled_classes = sorted(set(l for l in label_list if l != "unlabeled"))
        if labeled_mask.sum() > 1 and len(labeled_classes) > 1:
            km = KMeans(
                n_clusters=len(labeled_classes),
                random_state=self.seed,
                n_init="auto",
            ).fit(proj[labeled_mask])
            ri = adjusted_rand_score(
                [l for l in label_list if l != "unlabeled"],
                km.labels_,
            )
            ri_str = f"Rand Index: {ri:.2f}"
        else:
            ri_str = "Rand Index: n/a"

        title = f"{method.upper()} — CLS last layer | {mode.upper()}\n{ri_str}"

        unlabeled = np.array([l == "unlabeled" for l in label_list])
        if unlabeled.any():
            sns.scatterplot(
                x=proj[unlabeled, 0],
                y=proj[unlabeled, 1],
                color="gray",
                s=20,
                alpha=0.2,
                ax=ax,
                label="unlabeled",
                zorder=1,
                rasterized=True,
            )

        texts = []
        for lbl in sorted(set(label_list) - {"unlabeled"}):
            mask = np.array([l == lbl for l in label_list])
            size = 70 if lbl in highlight else 40
            alpha = 1.0 if lbl in highlight else 0.9
            sns.scatterplot(
                x=proj[mask, 0],
                y=proj[mask, 1],
                color=color_map[lbl],
                s=size,
                alpha=alpha,
                marker="o",
                ax=ax,
                label=lbl,
                zorder=2,
                rasterized=True,
            )
            if show_text:
                for x, y in proj[mask]:
                    texts.append(ax.text(
                        x,
                        y,
                        str(lbl),
                        fontsize=7,
                        alpha=0.85,
                        fontweight="bold" if lbl in highlight else "normal",
                        color=color_map.get(lbl, "black"),
                    ))

        if show_text and texts:
            adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

        ax.set_title(title)
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")

        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(
            handles,
            labels_,
            title=legend_title,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
        ).set_zorder(3)

    # ══════════════════════════════════════════════════════════════════════════
    #  run() — train + embed + plot
    # ══════════════════════════════════════════════════════════════════════════

    def run(
        self,
        methods: List[str] = ("umap", "tsne"),
        modes: List[str] = ("original", "TRDV", "TRDJ"),
    ) -> dict:
        """
        Run the full pipeline: train -> embed -> plot.

        All outputs are saved under target_dir.
        """
        print("=" * 60)
        print("[run] Step 1/3 — Training")
        print("=" * 60)
        history = self.train()

        print("=" * 60)
        print("[run] Step 2/3 — Embedding")
        print("=" * 60)
        df_emb = self.embed()

        print("=" * 60)
        print("[run] Step 3/3 — Plotting")
        print("=" * 60)
        plot_res = self.plot(
            df_embeddings=df_emb,
            methods=methods,
            modes=modes,
        )

        self._write_summary(
            stage="run",
            details={
                "message": "Full pipeline complete.",
                "methods": list(methods),
                "modes": list(modes),
                "model_file": self.save_path,
                "embedding_dir": self.embed_dir,
                "plot_dir": self.plot_dir,
            },
        )

        print("\nPipeline complete.")
        print(f"Summary saved to {self.summary_path}")
        return {
            "train_history": history,
            "df_embeddings": df_emb,
            "plot_results": plot_res,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Script entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pipe = ESMCPipeline(
        data_csv="/path/to/your/sequences.csv",
        target_dir="/path/to/output/",
        seed=42,
        epochs=5,
    )
    pipe.run()
