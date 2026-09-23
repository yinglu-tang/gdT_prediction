import os
import re
import time
import pickle
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, List, Optional

# ESM imports
from esm.utils.constants.models import ESMC_600M
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

# Default leiden columns used across the pipeline when none are specified
_DEFAULT_LEIDEN_COLS: List[str] = ["leiden_0.4"]


class TCRPipeline:
    def __init__(self, base_outdir: str, leiden_cols: Optional[List[str]] = None):
        """
        Parameters
        ----------
        base_outdir : str
            Directory where all pipeline outputs are written.
        leiden_cols : list of str, optional
            Leiden cluster column name(s) to carry through the pipeline.
            Defaults to ["leiden_0.4"] when not provided.
            Example: ["leiden_0.2", "leiden_0.4", "leiden_0.8"]
        """
        self.base_outdir = base_outdir
        self.leiden_cols: List[str] = leiden_cols if leiden_cols is not None else _DEFAULT_LEIDEN_COLS
        os.makedirs(self.base_outdir, exist_ok=True)
        print(f"Initialized TCR Pipeline. Outputs will be saved to: {self.base_outdir}")
        print(f"Leiden columns configured: {self.leiden_cols}")

    # ==========================================
    # 1. HELPERS
    # ==========================================
    @staticmethod
    def _coalesce_duplicate_columns(df: pd.DataFrame, col: str) -> pd.DataFrame:
        dup_mask = (df.columns == col)
        if dup_mask.sum() <= 1: return df
        sub = df.loc[:, dup_mask].copy()
        for c in sub.columns: sub[c] = sub[c].astype(object)
        merged = sub.bfill(axis=1).iloc[:, 0]
        df = df.loc[:, ~dup_mask].copy()
        df[col] = merged
        return df

    @staticmethod
    def _clean_garbage(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns: return df
        mask = pd.to_numeric(df[col], errors='coerce').notna()
        if mask.sum() > 0:
            print(f"Cleaning {mask.sum()} numeric garbage entries in {col}...")
            df.loc[mask, col] = np.nan
        return df

    @staticmethod
    def _get_cohort_id(s):
        if pd.isna(s): return pd.NA
        return re.sub(r"\d+$", "", str(s))

    # ==========================================
    # 2. PREPROCESSING
    # ==========================================
    def process_and_export_sequences(
        self,
        adata,
        cluster_col: Optional[str] = None,
        leiden_cols: Optional[List[str]] = None,
        batch_col: str = "batch",
        target_batch: str = "batch01",
        keep_clusters=None,
        merge_rules=None,
        out_prefix: str = "cohort3",
    ):
        """
        Preprocess AnnData, filter cells, assign clonotypes, and export CSVs.

        Parameters
        ----------
        adata : AnnData
            Input single-cell object.
        cluster_col : str, optional
            Primary cluster column used for filtering / merging (keep_clusters,
            merge_rules).  Falls back to the first entry of ``leiden_cols`` (or
            self.leiden_cols) when not provided.
        leiden_cols : list of str, optional
            All leiden columns to preserve in the exported CSV.  Defaults to
            ``self.leiden_cols``.  These columns are cast to str and carried
            through to the output so they are available for downstream steps.
        batch_col : str
            Column in adata.obs that identifies the batch.
        target_batch : str
            Value in batch_col to keep; pass None/empty to skip batch filtering.
        keep_clusters : list, optional
            Cluster labels (in cluster_col) to retain.
        merge_rules : list of (list, str), optional
            Each entry is ([old_labels], new_label) applied to cluster_col.
        out_prefix : str
            Prefix for all output file names.

        Returns
        -------
        tuple of str
            Paths to (main CSV, TRD unique CSV, TRG unique CSV).
        """
        # Resolve which leiden columns to use
        active_leiden_cols: List[str] = leiden_cols if leiden_cols is not None else self.leiden_cols

        # cluster_col drives filtering/merging; default to first active leiden col
        if cluster_col is None:
            cluster_col = active_leiden_cols[0]

        print("\n--- Starting Data Processing ---")
        print(f"  cluster_col (filtering/merging): {cluster_col}")
        print(f"  leiden_cols (preserved in output): {active_leiden_cols}")

        df = adata.obs.copy()
        df["cell_id"] = df.index.astype(str)

        # Batch Filtering
        if batch_col in df.columns and target_batch:
            df = df[df[batch_col] == target_batch].copy()

        # Cast key columns to str
        cols_to_cast = ["sample_id", "cohort_id", cluster_col] + active_leiden_cols
        for col in dict.fromkeys(cols_to_cast):          # deduplicate, preserve order
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Cluster Filtering & Merging (operates on cluster_col)
        if keep_clusters:
            df = df[df[cluster_col].isin(set(map(str, keep_clusters)))]

        df["merged_cluster"] = df[cluster_col].copy()
        if merge_rules:
            for old_list, new_label in merge_rules:
                mask = df[cluster_col].isin(map(str, old_list))
                df.loc[mask, "merged_cluster"] = str(new_label)

        # Clean TCR Data
        rename_map = {"trd_seq": "trd_sequence", "trg_seq": "trg_sequence"}
        df.rename(columns=rename_map, inplace=True)

        for col in ["trd_sequence", "trg_sequence"]:
            df = self._coalesce_duplicate_columns(df, col)
            df = self._clean_garbage(df, col)

        groupby_cols = ["TRDV", "TRDJ", "trd_sequence", "TRGV", "TRGJ", "trg_sequence"]
        for col in groupby_cols:
            if col not in df.columns:
                df[col] = "None"
            else:
                s = df[col]
                if isinstance(s, pd.DataFrame):
                    df = self._coalesce_duplicate_columns(df, col)
                    s = df[col]
                df[col] = s.astype(object).where(~pd.isna(s), "None").astype(str)

        # Clonotypes
        df["clone_size"] = df.groupby(groupby_cols)["cell_id"].transform("count")
        unique_clones = (
            df.groupby(groupby_cols)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        unique_clones["clonotype_id"] = [f"Clonotype_{i+1}" for i in range(len(unique_clones))]
        df = df.merge(unique_clones[groupby_cols + ["clonotype_id"]], on=groupby_cols, how="left")
        df["is_duplicated_clone"] = df["clone_size"] > 1

        # Save Main Filtered Data
        out_csv = os.path.join(self.base_outdir, f"{out_prefix}_final_tcr_merged.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved main processed data to: {out_csv}")

        # Extract & Save Unique Sequences for Embedding
        print("Extracting unique sequences for ESM...")
        df_trd = (
            df[["TRDV", "trd_sequence", "TRDJ"]]
            .rename(columns={"trd_sequence": "sequence"})
            .dropna(how="any")
        )
        df_trg = (
            df[["TRGV", "trg_sequence", "TRGJ"]]
            .rename(columns={"trg_sequence": "sequence"})
            .dropna(how="any")
        )

        df_trd = df_trd[~df_trd.isin(["NA", "nan", "None"]).any(axis=1)].drop_duplicates(subset=["sequence"])
        df_trg = df_trg[~df_trg.isin(["NA", "nan", "None"]).any(axis=1)].drop_duplicates(subset=["sequence"])

        trd_csv = os.path.join(self.base_outdir, f"{out_prefix}_trd_unique.csv")
        trg_csv = os.path.join(self.base_outdir, f"{out_prefix}_trg_unique.csv")
        df_trd.to_csv(trd_csv, index=False)
        df_trg.to_csv(trg_csv, index=False)

        return out_csv, trd_csv, trg_csv

    # ==========================================
    # 3. ESM EMBEDDINGS (CORRECTED)
    # ==========================================
    def run_esm_embeddings(self, seq_csv: str, model_weights: str, out_file: str, chunk_size=256):
        print(f"\n--- Running ESM Embeddings for {seq_csv} ---")
        df = pd.read_csv(seq_csv).dropna(subset=["sequence"])
        sequences = df["sequence"].tolist()

        model = ESMC.from_pretrained(ESMC_600M)
        model.load_state_dict(torch.load(model_weights, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()  # Ensure model is in evaluation mode

        config = LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True)

        def _embed(seq):
            prot = ESMProtein(sequence=seq)
            return model.logits(model.encode(prot), config)

        all_records = []
        total = len(sequences)

        # Process sequentially to avoid PyTorch thread collisions
        with torch.no_grad():  # Disable gradients for faster inference & less memory
            for i in range(0, total, chunk_size):
                chunk = sequences[i:i + chunk_size]

                for seq in chunk:
                    try:
                        out = _embed(seq)
                        if hasattr(out, "hidden_states"):
                            # Last hidden layer (shape: 1 x L x D), CLS token = position 0
                            cls_last = out.hidden_states[-1][0, 0, :].float().cpu()
                            all_records.append({"sequence": seq, "cls_last": cls_last})
                        else:
                            print(f"Skipping sequence due to missing hidden_states: {seq}")
                    except Exception as e:
                        print(f"Skipping sequence due to error: {seq}. Error: {e}")

                print(f"Processed {min(i + chunk_size, total)} / {total}")

                # Free memory after every chunk
                torch.cuda.empty_cache()

        # Save out the embeddings
        torch.save(all_records, out_file)

        # Clean up model from GPU memory
        del model
        torch.cuda.empty_cache()

        print(f"Saved embeddings to: {out_file}")
        return out_file

    # ==========================================
    # 4. TENSOR BUILDING (ESMC & ProBERT)
    # ==========================================
    def _load_lookup(self, path: str, emb_type: str):
        lookup = {}
        if emb_type.lower() == "esmc":
            data = torch.load(path)
            lookup = {rec["sequence"]: rec["cls_last"] for rec in data}
        elif emb_type.lower() == "probert":
            with open(path, "rb") as f:
                data = pickle.load(f)
            cls_np, info = data["cls_embeddings"], data["sample_info"]
            for i in range(len(info)):
                lookup[str(info[i][1])] = torch.tensor(cls_np[i], dtype=torch.float32)
        return lookup

    def build_tensors(
        self,
        tcr_csv: str,
        trd_emb_path: str,
        trg_emb_path: str,
        emb_type: str,
        out_prefix: str,
        leiden_cols: Optional[List[str]] = None,
    ):
        """
        Build per-cell embedding tensors and save them to disk.

        Parameters
        ----------
        tcr_csv : str
            Path to the main processed CSV (output of process_and_export_sequences).
        trd_emb_path : str
            Path to the TRD embedding file (.pt for ESMC, .pkl for ProBERT).
        trg_emb_path : str
            Path to the TRG embedding file.
        emb_type : str
            One of "esmc" or "probert".
        out_prefix : str
            Prefix for all output .pt file names.
        leiden_cols : list of str, optional
            Leiden column name(s) to include in every saved record's metadata.
            Defaults to ``self.leiden_cols``.
            Columns missing from the CSV are stored as "N/A".

        Returns
        -------
        list of dict
            The concatenated (TRD + TRG) embedding records.
        """
        # Resolve leiden columns for this call
        active_leiden_cols: List[str] = leiden_cols if leiden_cols is not None else self.leiden_cols

        print(f"\n--- Building Final Tensors ({emb_type}) for {tcr_csv} ---")
        print(f"  leiden_cols included in metadata: {active_leiden_cols}")

        df = pd.read_csv(tcr_csv, header=0)
        df = self._clean_garbage(self._clean_garbage(df, "trd_sequence"), "trg_sequence")
        df["cohort_id"] = df["sample_id"].apply(self._get_cohort_id)
        df["unique_id"] = df["cell_id"].astype(str) + "_" + df["sample_id"].astype(str)

        trd_lookup = self._load_lookup(trd_emb_path, emb_type)
        trg_lookup = self._load_lookup(trg_emb_path, emb_type)

        emb_dim = (
            len(list(trd_lookup.values())[0]) if trd_lookup
            else (len(list(trg_lookup.values())[0]) if trg_lookup else 768)
        )

        df["trd_emb"] = df["trd_sequence"].apply(
            lambda s: trd_lookup.get(s, None) if isinstance(s, str) else None
        )
        df["trg_emb"] = df["trg_sequence"].apply(
            lambda s: trg_lookup.get(s, None) if isinstance(s, str) else None
        )

        # Averages
        s_trd, s_trg, c_trd, c_trg = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for _, r in df.iterrows():
            if r["trd_emb"] is not None:
                s_trd[r["sample_id"]].append(r["trd_emb"])
                c_trd[r["cohort_id"]].append(r["trd_emb"])
            if r["trg_emb"] is not None:
                s_trg[r["sample_id"]].append(r["trg_emb"])
                c_trg[r["cohort_id"]].append(r["trg_emb"])

        s_avg_trd = {k: torch.stack(v).mean(0) for k, v in s_trd.items()}
        s_avg_trg = {k: torch.stack(v).mean(0) for k, v in s_trg.items()}
        c_avg_trd = {k: torch.stack(v).mean(0) for k, v in c_trd.items()}
        c_avg_trg = {k: torch.stack(v).mean(0) for k, v in c_trg.items()}
        g_avg_trd = torch.stack(list(s_avg_trd.values())).mean(0) if s_avg_trd else torch.zeros(emb_dim)
        g_avg_trg = torch.stack(list(s_avg_trg.values())).mean(0) if s_avg_trg else torch.zeros(emb_dim)

        # Resolution + record building
        trd_list, trg_list, concat_list = [], [], []
        for _, row in df.iterrows():
            sid, cid = row["sample_id"], row["cohort_id"]

            t_trd = row["trd_emb"]
            if t_trd is None:
                t_trd = s_avg_trd.get(sid, c_avg_trd.get(cid, g_avg_trd))

            t_trg = row["trg_emb"]
            if t_trg is None:
                t_trg = s_avg_trg.get(sid, c_avg_trg.get(cid, g_avg_trg))

            # Build metadata dict with all requested leiden columns
            leiden_meta = {col: row.get(col, "N/A") for col in active_leiden_cols}

            meta = {
                "unique_id": row["unique_id"],
                "sample_id": row["sample_id"],
                "cohort_id": row["cohort_id"],
                "cell_id": row["cell_id"],
                **leiden_meta,
            }

            trd_list.append({
                **meta,
                "TRDV": row["TRDV"], "TRDJ": row["TRDJ"],
                "sequence": row["trd_sequence"],
                "cls_last": t_trd,
            })
            trg_list.append({
                **meta,
                "TRGV": row["TRGV"], "TRGJ": row["TRGJ"],
                "sequence": row["trg_sequence"],
                "cls_last": t_trg,
            })
            concat_list.append({
                **meta,
                "TRDV": row["TRDV"], "TRDJ": row["TRDJ"], "trd_sequence": row["trd_sequence"],
                "TRGV": row["TRGV"], "TRGJ": row["TRGJ"], "trg_sequence": row["trg_sequence"],
                "cls_last": torch.cat([t_trd, t_trg], dim=0),
            })

        torch.save(trd_list,    os.path.join(self.base_outdir, f"{out_prefix}_trd_{emb_type}.pt"))
        torch.save(trg_list,    os.path.join(self.base_outdir, f"{out_prefix}_trg_{emb_type}.pt"))
        torch.save(concat_list, os.path.join(self.base_outdir, f"{out_prefix}_concat_{emb_type}.pt"))
        print(f"Finished building tensors for {out_prefix}.")
        return concat_list