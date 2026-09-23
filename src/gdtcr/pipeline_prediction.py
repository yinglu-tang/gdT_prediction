"""TCR embedding extraction and ESMC, ProBERT, V/J, and fused prediction."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_LEIDEN_COLS: List[str] = ["leiden_0.4"]
_PREDICTION_MODES = ("esmc", "probert", "vj", "fused")

def _validate_prediction_mode(emb_type: str) -> str:
    mode = str(emb_type).lower()
    if mode not in _PREDICTION_MODES:
        raise ValueError(f"Unknown emb_type={emb_type!r}. Choose from {_PREDICTION_MODES}.")
    return mode

def _patch_sympy_printing_compat() -> None:
    """Expose StrPrinter for older TAPE dependencies."""
    try:
        import importlib
        import sympy  # type: ignore

        printing_mod = importlib.import_module("sympy.printing")

        if not hasattr(sympy, "printing"):
            setattr(sympy, "printing", printing_mod)

        if not hasattr(printing_mod, "StrPrinter"):
            try:
                from sympy.printing.str import StrPrinter  # type: ignore
                setattr(printing_mod, "StrPrinter", StrPrinter)
            except Exception:
                pass
    except Exception:
        pass

_patch_sympy_printing_compat()

try:
    import scanpy as sc
    _HAS_SCANPY = True
except ImportError:
    _HAS_SCANPY = False

try:
    from .pipeline_TCR import TCRPipeline
    _HAS_TCR = True
except ImportError as e:
    _HAS_TCR = False
    warnings.warn(
        f"ESMC dependencies unavailable ({e}). Install gdtcr[esmc] to enable this backend.",
        ImportWarning,
        stacklevel=2,
    )

try:
    from .pipeline_probert import run_eval as _probert_run_eval
    _HAS_PROBERT = True
except Exception as e:
    _probert_run_eval = None
    _HAS_PROBERT = False
    warnings.warn(
        "pipeline_probert not importable (likely wrong conda env or SymPy/TAPE "
        f"compatibility issue: {type(e).__name__}: {e}) – ProBERT steps will be skipped.",
        ImportWarning,
        stacklevel=2,
    )

class EmbeddingPipeline:
    """Build TCR embeddings and predict clusters with ESMC, ProBERT, V/J, or fusion."""

    def __init__(
        self,
        output_root:        str | Path,
        esmc_trd_weights:   str | Path = "",
        esmc_trg_weights:   str | Path = "",
        probert_trd_model:  str | Path = "",
        probert_trg_model:  str | Path = "",
        probert_base_path:  str | Path = "../models/proteinbert",
        leiden_cols:        Optional[List[str]] = None,
        esmc_chunk_size:    int = 1000,
        probert_batch_size: int = 64,
        probert_max_length: int = 45,
    ):
        self.output_root = Path(output_root)

        self.esmc_trd_weights  = str(esmc_trd_weights)
        self.esmc_trg_weights  = str(esmc_trg_weights)
        self.probert_trd_model = str(probert_trd_model)
        self.probert_trg_model = str(probert_trg_model)
        self.probert_base_path = str(probert_base_path)

        self.leiden_cols: List[str] = leiden_cols if leiden_cols is not None else _DEFAULT_LEIDEN_COLS

        self.esmc_chunk_size    = esmc_chunk_size
        self.probert_batch_size = probert_batch_size
        self.probert_max_length = probert_max_length

        self._tcr = (
            TCRPipeline(base_outdir=str(self.output_root), leiden_cols=self.leiden_cols)
            if _HAS_TCR else None
        )

        self.output_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,

        h5ad_path:     Optional[str | Path] = None,
        cluster_col:   Optional[str] = None,
        batch_col:     str = "batch",
        target_batch:  str = "batch01",
        keep_clusters: Optional[List[str]] = None,
        merge_rules:   Optional[List[Tuple]] = None,

        tcr_csv: Optional[str | Path] = None,

        out_prefix:  str  = "cohort3",
        run_esmc:    bool = True,
        run_probert: bool = True,

        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
        id_col: Optional[str] = None,  # backward-compatible alias for cell_id_col
        reference: bool = False,

        leiden_cols: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Build embeddings from either an annotated AnnData file or a TCR CSV."""
        if h5ad_path is None and tcr_csv is None:
            raise ValueError("Provide either h5ad_path or tcr_csv.")
        if h5ad_path is not None and tcr_csv is not None:
            raise ValueError("Provide h5ad_path OR tcr_csv, not both.")

        active_leiden_cols: List[str] = leiden_cols if leiden_cols is not None else self.leiden_cols

        if cluster_col is None:
            cluster_col = active_leiden_cols[0]

        if cell_id_col is None and id_col is not None:
            cell_id_col = id_col

        print(f"\n  leiden_cols   : {active_leiden_cols}")
        print(f"  cluster_col   : {cluster_col}")
        print(f"  cell_id_col   : {cell_id_col}")
        print(f"  sample_id_col : {sample_id_col}")
        print(f"  dataset_col   : {dataset_col}")
        print(f"  reference     : {reference}")

        results: Dict[str, object] = {
            "tcr_csv": None, "computed_tcr_csv": None, "trd_seq_csv": None, "trg_seq_csv": None,
            "esmc_concat": None, "probert_concat": None,
        }

        if h5ad_path is not None:
            tcr_csv, trd_seq_csv, trg_seq_csv = self._step1_from_h5ad(
                h5ad_path, cluster_col, batch_col, target_batch,
                keep_clusters or [], merge_rules or [], out_prefix,
                active_leiden_cols,
            )
        else:
            tcr_csv, trd_seq_csv, trg_seq_csv = self._step1_from_csv(
                str(tcr_csv), out_prefix,
                cell_id_col=cell_id_col,
                sample_id_col=sample_id_col,
                dataset_col=dataset_col,
                reference=reference,
            )

        results.update(
            tcr_csv=tcr_csv, computed_tcr_csv=tcr_csv, trd_seq_csv=trd_seq_csv, trg_seq_csv=trg_seq_csv
        )

        if run_esmc:
            results["esmc_concat"] = self._run_esmc(
                tcr_csv, trd_seq_csv, trg_seq_csv, out_prefix,
                active_leiden_cols,
                cell_id_col="cell_id",                  # computed CSV always has this
                sample_id_col="sample_id",              # computed CSV always has this
                dataset_col="dataset",                  # computed CSV always has this
            )

        if run_probert:
            results["probert_concat"] = self._run_probert(
                tcr_csv, trd_seq_csv, trg_seq_csv, out_prefix,
                active_leiden_cols,
                cell_id_col="cell_id",                  # computed CSV always has this
                sample_id_col="sample_id",              # computed CSV always has this
                dataset_col="dataset",                  # computed CSV always has this
            )

        print("\n=== All done ===")
        print(f"  All outputs saved to: {self.output_root}")
        return results

    def _step1_from_h5ad(
        self,
        h5ad_path, cluster_col, batch_col, target_batch,
        keep_clusters, merge_rules, out_prefix,
        leiden_cols: List[str],
    ) -> Tuple[str, str, str]:
        print("\n=== STEP 1: Load h5ad and export unique sequences ===")

        if not _HAS_SCANPY:
            raise ImportError("scanpy is required to load .h5ad files.")
        if self._tcr is None:
            raise ImportError("pipeline_TCR is required for this step.")

        adata = sc.read_h5ad(str(h5ad_path))

        tcr_csv, trd_seq_csv, trg_seq_csv = self._tcr.process_and_export_sequences(
            adata,
            cluster_col   = cluster_col,
            leiden_cols   = leiden_cols,
            batch_col     = batch_col,
            target_batch  = target_batch,
            keep_clusters = keep_clusters,
            merge_rules   = merge_rules,
            out_prefix    = out_prefix,
        )
        print(f"  TCR CSV     : {tcr_csv}")
        print(f"  TRD seq CSV : {trd_seq_csv}")
        print(f"  TRG seq CSV : {trg_seq_csv}")
        return tcr_csv, trd_seq_csv, trg_seq_csv

    @staticmethod
    def _bad_sequence_values() -> set:
        return {"", "NA", "NaN", "nan", "None", "none", "NULL", "null", "<NA>"}

    @classmethod
    def _valid_seq_series(cls, s):
        z = s.astype("string").str.strip()
        return (~z.isna()) & (~z.isin(cls._bad_sequence_values()))

    def _make_unique_chain_sequence_csv(
        self,
        df,
        out_prefix: str,
        chain_label: str,
        seq_col_in: str,
        v_col: str,
        j_col: str,
    ) -> str:
        """Save one row per unique valid CDR3 sequence for one chain."""
        import pandas as pd

        for c in [seq_col_in, v_col, j_col]:
            if c not in df.columns:
                df[c] = "None"

        work = df[[seq_col_in, v_col, j_col]].copy()
        work["sequence"] = work[seq_col_in].astype("string").str.strip()
        valid = self._valid_seq_series(work["sequence"])
        n_cells_valid = int(valid.sum())

        work = work.loc[valid, ["sequence", v_col, j_col]].copy()

        work[v_col] = work[v_col].fillna("None").astype(str).str.strip()
        work[j_col] = work[j_col].fillna("None").astype(str).str.strip()

        unique = (
            work.groupby("sequence", as_index=False)
                .agg(**{
                    v_col: (v_col, "first"),
                    j_col: (j_col, "first"),
                    "n_cells": ("sequence", "size"),
                })
                .sort_values(["n_cells", "sequence"], ascending=[False, True])
                .reset_index(drop=True)
        )

        out_csv = str(self.output_root / f"{out_prefix}_{chain_label.lower()}_unique_sequences.csv")
        unique.to_csv(out_csv, index=False)

        print(f"  {chain_label} unique sequence CSV: {out_csv}")
        print(f"    valid cells: {n_cells_valid:,} / {len(df):,}")
        print(f"    unique valid sequences: {len(unique):,}")
        if len(unique):
            print(f"    first saved rows:\n{unique.head(5).to_string(index=False)}")
        else:
            print(f"    [WARN] No valid {chain_label} sequences found.")
        return out_csv

    @staticmethod
    def _is_na_like_value(x: Any) -> bool:
        if x is None:
            return True
        s = str(x).strip()
        low = s.lower()
        up = s.upper()
        return s == "" or low in {"na", "nan", "none", "null", "<na>"} or up.startswith("NA_")

    @classmethod
    def _reference_vj_value(cls, x: Any) -> Any:
        """For reference mode: keep NA-like values; keep alleles; add *01 to non-NA genes without allele."""
        if cls._is_na_like_value(x):
            return x
        s = str(x).strip()
        if "*" in s:
            return s
        return f"{s}*01"

    def _resolve_column(self, df, requested: Optional[str], candidates: List[str], role: str) -> Optional[str]:
        """Resolve a metadata column by explicit name first, then candidate names."""
        if requested is not None:
            if requested not in df.columns:
                raise ValueError(
                    f"{role}={requested!r} not found in input CSV. "
                    f"Available columns include: {list(df.columns)[:80]}"
                )
            return requested
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None

    def _prepare_identifier_columns(
        self,
        df,
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
    ):
        """Resolve unique cell identifiers and sample/dataset metadata."""
        n = len(df)

        resolved_cell_col = self._resolve_column(
            df, cell_id_col, candidates=["cell_id", "barcode"], role="cell_id_col"
        )
        if resolved_cell_col is None:
            resolved_cell_col = "__row_id__"
            df[resolved_cell_col] = [f"row_{i}" for i in range(n)]

        resolved_sample_col = self._resolve_column(
            df, sample_id_col, candidates=["sample_id", "dataset"], role="sample_id_col"
        )
        if resolved_sample_col is None:

            resolved_sample_col = resolved_cell_col

        resolved_dataset_col = self._resolve_column(
            df, dataset_col, candidates=["dataset"], role="dataset_col"
        )
        if resolved_dataset_col is None:
            resolved_dataset_col = resolved_sample_col

        df["cell_id"] = df[resolved_cell_col].astype(str)
        df["sample_id"] = df[resolved_sample_col].astype(str)
        df["dataset"] = df[resolved_dataset_col].astype(str)

        n_dup = int(df["cell_id"].duplicated().sum())
        if n_dup > 0:
            print(
                f"  [WARN] cell_id has {n_dup:,} duplicated rows after resolving from "
                f"{resolved_cell_col!r}. cell_id should uniquely identify each entry."
            )

        return df, {
            "cell_id_col": resolved_cell_col,
            "sample_id_col": resolved_sample_col,
            "dataset_col": resolved_dataset_col,
        }

    def _step1_from_csv(
        self,
        tcr_csv: str,
        out_prefix: str,
        id_col: Optional[str] = None,
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
        reference: bool = False,
    ) -> Tuple[str, str, str]:
        """Prepare identifiers, optional V/J alleles, and unique chain sequence tables."""
        print("\n=== STEP 1: Prepare metadata and save unique TRD/TRG sequences ===")
        import pandas as pd

        if cell_id_col is None and id_col is not None:
            cell_id_col = id_col

        df = pd.read_csv(tcr_csv)
        df, id_sources = self._prepare_identifier_columns(
            df, cell_id_col=cell_id_col, sample_id_col=sample_id_col, dataset_col=dataset_col
        )

        for c in ["trd_sequence", "TRDV", "TRDJ", "trg_sequence", "TRGV", "TRGJ"]:
            if c not in df.columns:
                df[c] = "None"

        if reference:
            vj_cols = ["TRDV", "TRGV", "TRDJ", "TRGJ"]
            before = {c: df[c].copy() for c in vj_cols}
            for c in vj_cols:
                df[c] = df[c].map(self._reference_vj_value)
            print("  Reference V/J allele normalization enabled:")
            for c in vj_cols:
                changed = int((before[c].astype(str) != df[c].astype(str)).sum())
                print(f"    {c}: added/changed allele format in {changed:,} rows")

        computed_csv = str(self.output_root / f"{out_prefix}_computed_metadata.csv")
        df.to_csv(computed_csv, index=False)

        print(f"  Input metadata CSV     : {tcr_csv}")
        print(f"  cell_id source column  : {id_sources['cell_id_col']}")
        print(f"  sample_id source column: {id_sources['sample_id_col']}")
        print(f"  dataset source column  : {id_sources['dataset_col']}")
        print(f"  Computed metadata CSV  : {computed_csv}")
        print(f"  First computed rows:")
        preview_cols = [c for c in ["sample_id", "cell_id", "dataset", "TRDV", "TRDJ", "trd_sequence", "TRGV", "TRGJ", "trg_sequence"] if c in df.columns]
        print(df[preview_cols].head(5).to_string(index=False))

        trd_seq_csv = self._make_unique_chain_sequence_csv(
            df=df, out_prefix=out_prefix, chain_label="TRD",
            seq_col_in="trd_sequence", v_col="TRDV", j_col="TRDJ",
        )
        trg_seq_csv = self._make_unique_chain_sequence_csv(
            df=df, out_prefix=out_prefix, chain_label="TRG",
            seq_col_in="trg_sequence", v_col="TRGV", j_col="TRGJ",
        )

        print(f"  Metadata CSV for concat build: {computed_csv}")
        return computed_csv, trd_seq_csv, trg_seq_csv

    def _run_esmc(
        self,
        tcr_csv: str,
        trd_seq_csv: str,
        trg_seq_csv: str,
        out_prefix: str,
        leiden_cols: List[str],
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
    ):
        if self._tcr is None:
            print("\n[SKIP] ESMC – pipeline_TCR not importable in this env.")
            return None

        print("\n=== STEP 2: ESMC embeddings from UNIQUE sequences ===")
        trd_pt = str(self.output_root / f"{out_prefix}_trd_esmc_embeddings.pt")
        trg_pt = str(self.output_root / f"{out_prefix}_trg_esmc_embeddings.pt")

        import pandas as pd
        import torch

        def _run_one(seq_csv: str, weights: str, out_file: str, chain: str) -> None:
            n_unique = int(len(pd.read_csv(seq_csv)))
            if n_unique == 0:
                torch.save({}, out_file)
                print(f"  [WARN] {chain} ESMC: no valid unique sequences; saved empty dict -> {out_file}")
                return
            print(f"  {chain} ESMC input: {n_unique:,} unique sequences from {seq_csv}")
            self._tcr.run_esm_embeddings(
                seq_csv=seq_csv,
                model_weights=weights,
                out_file=out_file,
                chunk_size=self.esmc_chunk_size,
            )

            seq_map = self._load_embedding_map(out_file)
            torch.save(seq_map, out_file)

        _run_one(trd_seq_csv, self.esmc_trd_weights, trd_pt, "TRD")
        _run_one(trg_seq_csv, self.esmc_trg_weights, trg_pt, "TRG")

        print("\n=== STEP 3: Build ESMC concat tensors from metadata CSV ===")
        concat = self._build_sequence_keyed_concat_tensors(
            tcr_csv=tcr_csv,
            trd_emb_path=trd_pt,
            trg_emb_path=trg_pt,
            emb_type="esmc",
            out_prefix=out_prefix,
            leiden_cols=leiden_cols,
            cell_id_col=cell_id_col,
            sample_id_col=sample_id_col,
            dataset_col=dataset_col,
        )
        print(f"  ESMC tensors -> {self.output_root}")
        return concat

    def _run_probert(
        self,
        tcr_csv: str,
        trd_seq_csv: str,
        trg_seq_csv: str,
        out_prefix: str,
        leiden_cols: List[str],
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
    ):
        if not _HAS_PROBERT:
            print("\n[SKIP] ProBERT – pipeline_probert not importable in this env.")
            return None

        print("\n=== STEP 4: ProBERT embeddings ===")
        trd_pkl = str(self.output_root / f"{out_prefix}_trd_probert.pkl")
        trg_pkl = str(self.output_root / f"{out_prefix}_trg_probert.pkl")

        self._probert_embed(trd_seq_csv, "sequence", self.probert_trd_model, trd_pkl, "TRD")
        self._probert_embed(trg_seq_csv, "sequence", self.probert_trg_model, trg_pkl, "TRG")

        print("\n=== STEP 5: Build ProBERT concat tensors from metadata CSV ===")
        concat = self._build_sequence_keyed_concat_tensors(
            tcr_csv=tcr_csv,
            trd_emb_path=trd_pkl,
            trg_emb_path=trg_pkl,
            emb_type="probert",
            out_prefix=out_prefix,
            leiden_cols=leiden_cols,
            cell_id_col=cell_id_col,
            sample_id_col=sample_id_col,
            dataset_col=dataset_col,
        )
        print(f"  ProBERT tensors -> {self.output_root}")
        return concat

    def _probert_embed(
        self,
        data_path: str,
        seq_col: str,
        model_path: str,
        output_pkl: str,
        chain_label: str = "",
    ) -> None:
        """Save sequence-keyed CLS embeddings from a fine-tuned ProBERT model."""
        import pickle
        import numpy as np
        import pandas as pd

        df_raw = pd.read_csv(data_path)

        if seq_col not in df_raw.columns:
            with open(output_pkl, "wb") as f:
                pickle.dump({}, f)
            print(f"  [WARN] {chain_label} ProBERT: column '{seq_col}' not found "
                  f"in {data_path}; chain will be zero-filled.")
            return

        _bad = {"", "NA", "NaN", "nan", "None", "none", "NULL", "null"}
        seqs = df_raw[seq_col].astype(str).str.strip()
        valid_mask = seqs.notna() & ~seqs.isin(_bad)
        unique_seqs = seqs[valid_mask].drop_duplicates().reset_index(drop=True)

        if len(unique_seqs) == 0:
            with open(output_pkl, "wb") as f:
                pickle.dump({}, f)
            print(f"  [WARN] {chain_label} ProBERT: no valid sequences; "
                  f"chain will be zero-filled.")
            return

        clean_csv = str(Path(output_pkl).with_suffix("")) + "_unique_seqs.csv"
        unique_seqs.to_frame(name=seq_col).to_csv(clean_csv, index=False)
        print(f"  {chain_label} ProBERT input: {len(unique_seqs):,} unique sequences -> {clean_csv}")

        if _probert_run_eval is None:
            raise ImportError(
                "pipeline_probert.run_eval is not available in this environment."
            )

        eval_tmp = str(self.output_root / f"probert_{chain_label.lower()}_eval")
        eval_args = argparse.Namespace(
            data_path        = clean_csv,
            seq_col          = seq_col,
            batch_size       = self.probert_batch_size,
            max_length       = self.probert_max_length,
            tokenizer        = "unirep",
            proteinbert_path = self.probert_base_path,
            model_path       = model_path,
            output_path      = eval_tmp,
        )
        _probert_run_eval(eval_args)

        src_pkl = Path(eval_tmp) / "embeddings_eval.pkl"
        with open(src_pkl, "rb") as f:
            raw = pickle.load(f)

        if not isinstance(raw, dict) or "cls_embeddings" not in raw or "sample_info" not in raw:
            keys = list(raw.keys()) if isinstance(raw, dict) else "N/A"
            raise ValueError(
                f"{chain_label} ProBERT: unexpected pkl format in {src_pkl}. "
                f"type={type(raw)}, keys={keys}. "
                "Expected keys: cls_embeddings, sample_info, columns."
            )

        cls_embs = np.asarray(raw["cls_embeddings"])   # (N, dim)
        sample_info_raw = raw["sample_info"]
        columns = list(raw.get("columns", []))

        if len(cls_embs) != len(sample_info_raw):
            raise ValueError(
                f"{chain_label} ProBERT: cls_embeddings has {len(cls_embs)} rows "
                f"but sample_info has {len(sample_info_raw)} rows in {src_pkl}."
            )

        def _get_seq_from_info(info_row: Any) -> str:

            if isinstance(info_row, dict):
                for candidate in [seq_col, "sequence", "seq", "cdr3"]:
                    if candidate in info_row:
                        return str(info_row[candidate]).strip()
                return ""
            if columns:
                for candidate in [seq_col, "sequence", "seq", "cdr3"]:
                    if candidate in columns:
                        idx = columns.index(candidate)
                        try:
                            return str(info_row[idx]).strip()
                        except Exception:
                            return ""

            if isinstance(info_row, (str, bytes)):
                return str(info_row).strip()
            return ""

        import torch
        seq_to_emb: Dict[str, Any] = {}
        for info_row, emb_row in zip(sample_info_raw, cls_embs):
            seq = _get_seq_from_info(info_row)
            if seq and seq not in _bad:
                seq_to_emb[seq] = torch.tensor(emb_row, dtype=torch.float32)

        if len(seq_to_emb) == 0:
            raise ValueError(
                f"{chain_label} ProBERT: parsed zero sequence-keyed embeddings from {src_pkl}. "
                f"sample_info example={sample_info_raw[0] if len(sample_info_raw) else 'EMPTY'}, "
                f"columns={columns}"
            )

        with open(output_pkl, "wb") as f:
            pickle.dump(seq_to_emb, f)
        print(f"  {chain_label} ProBERT embeddings: {len(seq_to_emb):,} unique sequences → {output_pkl}")

    def _build_sequence_keyed_concat_tensors(
        self,
        tcr_csv: str | Path,
        trd_emb_path: str | Path,
        trg_emb_path: str | Path,
        emb_type: str,
        out_prefix: str,
        leiden_cols: Optional[List[str]] = None,
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
        id_col: Optional[str] = None,
    ) -> List[dict]:
        """Build one concat record per cell_id from sequence-keyed TRD/TRG embeddings."""
        import numpy as np
        import pandas as pd
        import torch

        if cell_id_col is None and id_col is not None:
            cell_id_col = id_col
        leiden_cols = list(leiden_cols or [])
        df = pd.read_csv(tcr_csv)

        for c in ["trd_sequence", "TRDV", "TRDJ", "trg_sequence", "TRGV", "TRGJ"]:
            if c not in df.columns:
                df[c] = "None"
        df, id_sources = self._prepare_identifier_columns(
            df, cell_id_col=cell_id_col, sample_id_col=sample_id_col, dataset_col=dataset_col
        )

        trd_seq_map = self._load_seq_embedding_map(trd_emb_path)
        trg_seq_map = self._load_seq_embedding_map(trg_emb_path)

        n_trd = sum(1 for s in df["trd_sequence"].astype(str).str.strip() if s in trd_seq_map)
        n_trg = sum(1 for s in df["trg_sequence"].astype(str).str.strip() if s in trg_seq_map)
        print(f"  Building {emb_type} concat from: {tcr_csv}")
        print(f"  cell_id source column  : {id_sources['cell_id_col']}")
        print(f"  sample_id source column: {id_sources['sample_id_col']}")
        print(f"  dataset source column  : {id_sources['dataset_col']}")
        print(f"  TRD matches: {n_trd:,} / {len(df):,}")
        print(f"  TRG matches: {n_trg:,} / {len(df):,}")

        if n_trd == 0 and n_trg == 0:
            raise ValueError(
                f"No TRD or TRG sequences in {tcr_csv} matched embedding files. "
                "Check trd_sequence/trg_sequence strings and embedding inputs."
            )

        any_trd = next(iter(trd_seq_map.values()), None)
        any_trg = next(iter(trg_seq_map.values()), None)
        trd_dim = self._to_1d_tensor(any_trd).shape[0] if any_trd is not None else (self._to_1d_tensor(any_trg).shape[0] if any_trg is not None else 1)
        trg_dim = self._to_1d_tensor(any_trg).shape[0] if any_trg is not None else trd_dim

        records: List[dict] = []
        X_rows = []
        meta_rows = []
        miss_rows = []

        for i, row in df.iterrows():
            cell_id = str(row["cell_id"])
            sample_id = str(row["sample_id"])
            dataset = str(row["dataset"])
            trd_seq = str(row.get("trd_sequence", "")).strip()
            trg_seq = str(row.get("trg_sequence", "")).strip()

            if trd_seq in trd_seq_map:
                trd_emb = self._to_1d_tensor(trd_seq_map[trd_seq]).float().cpu()
                trd_source = "observed"
            else:
                trd_emb = torch.zeros(trd_dim, dtype=torch.float32)
                trd_source = "missing"

            if trg_seq in trg_seq_map:
                trg_emb = self._to_1d_tensor(trg_seq_map[trg_seq]).float().cpu()
                trg_source = "observed"
            else:
                trg_emb = torch.zeros(trg_dim, dtype=torch.float32)
                trg_source = "missing"

            fused = torch.cat([trd_emb, trg_emb], dim=0).float().cpu()
            rec = {
                "cell_id": cell_id,
                "sample_id": sample_id,
                "row_index": int(i),
                "dataset": dataset,
                "trd_sequence": trd_seq,
                "trg_sequence": trg_seq,
                "TRDV": str(row.get("TRDV", "None")),
                "TRDJ": str(row.get("TRDJ", "None")),
                "TRGV": str(row.get("TRGV", "None")),
                "TRGJ": str(row.get("TRGJ", "None")),
                "trd_source": trd_source,
                "trg_source": trg_source,
                "trd_emb": trd_emb,
                "trg_emb": trg_emb,
                "embedding": fused,
                "x": fused,
                "cls_last": fused,
            }
            for col in leiden_cols:
                if col in row.index:
                    rec[col] = row[col]
            for col in df.columns:
                if col not in rec:
                    val = row[col]
                    if not isinstance(val, (list, tuple, dict)):
                        rec[col] = val

            records.append(rec)
            X_rows.append(fused.numpy())
            meta_rows.append({k: v for k, v in rec.items() if k not in {"trd_emb", "trg_emb", "embedding", "x", "cls_last"}})
            miss_rows.append({
                "row_index": int(i),
                "cell_id": cell_id,
                "sample_id": sample_id,
                "dataset": dataset,
                "trd_sequence": trd_seq,
                "trg_sequence": trg_seq,
                "trd_source": trd_source,
                "trg_source": trg_source,
                "trd_was_missing": trd_source == "missing",
                "trg_was_missing": trg_source == "missing",
            })

        X = np.vstack(X_rows).astype("float32")
        meta = pd.DataFrame(meta_rows)
        miss = pd.DataFrame(miss_rows)

        concat_path = str(self.output_root / f"{out_prefix}_{emb_type}_concat.pt")
        x_path = str(self.output_root / f"{out_prefix}_{emb_type}_X.npy")
        meta_path = str(self.output_root / f"{out_prefix}_{emb_type}_metadata.csv")
        miss_path = str(self.output_root / f"{out_prefix}_{emb_type}_missing_chain_handling.csv")
        miss_summary_path = str(self.output_root / f"{out_prefix}_{emb_type}_missing_chain_summary_by_sample.csv")

        miss_summary = (
            miss.groupby(["dataset", "sample_id"], dropna=False)
            .agg(
                n_cells=("cell_id", "count"),
                n_trd_missing=("trd_was_missing", "sum"),
                n_trg_missing=("trg_was_missing", "sum"),
            )
            .reset_index()
        )
        miss_summary["frac_trd_missing"] = miss_summary["n_trd_missing"] / miss_summary["n_cells"].clip(lower=1)
        miss_summary["frac_trg_missing"] = miss_summary["n_trg_missing"] / miss_summary["n_cells"].clip(lower=1)

        torch.save(records, concat_path)
        np.save(x_path, X)
        meta.to_csv(meta_path, index=False)
        miss.to_csv(miss_path, index=False)
        miss_summary.to_csv(miss_summary_path, index=False)

        print(f"  Built {emb_type} concat records: {len(records):,}")
        print(f"  X matrix: {X.shape} -> {x_path}")
        print(f"  Concat -> {concat_path}")
        print(f"  Metadata -> {meta_path}")
        print(f"  Missing-chain log -> {miss_path}")
        print(f"  Missing-chain summary by sample -> {miss_summary_path}")
        print("  First concat metadata rows:")
        show_cols = [c for c in ["sample_id", "cell_id", "dataset", "TRDV", "TRDJ", "trd_sequence", "TRGV", "TRGJ", "trg_sequence", "trd_source", "trg_source"] if c in meta.columns]
        print(meta[show_cols].head(5).to_string(index=False))
        return records

    def run_unlabeled(
        self,
        tcr_csv: Optional[str | Path] = None,
        h5ad_path: Optional[str | Path] = None,
        out_prefix: str = "newdata",
        run_esmc: bool = True,
        run_probert: bool = True,
        esmc_trd_emb_path: Optional[str | Path] = None,
        esmc_trg_emb_path: Optional[str | Path] = None,
        probert_trd_emb_path: Optional[str | Path] = None,
        probert_trg_emb_path: Optional[str | Path] = None,
        predict_model_path: Optional[str | Path] = None,
        predict_emb_type: str = "esmc",
        device: Optional[str] = None,
        id_col: Optional[str] = None,
        alpha: Optional[float] = None,
        prediction_batch_size: int = 512,
    ) -> Dict[str, object]:
        """Build or reuse embeddings for unlabeled records and optionally predict clusters."""
        if predict_model_path is not None:
            predict_emb_type = _validate_prediction_mode(predict_emb_type)
        if tcr_csv is None and h5ad_path is None:
            raise ValueError("Provide either tcr_csv or h5ad_path for unlabeled data.")
        if tcr_csv is not None and h5ad_path is not None:
            raise ValueError("Provide tcr_csv OR h5ad_path, not both.")

        if h5ad_path is not None:
            tcr_csv = self._unlabeled_csv_from_h5ad(h5ad_path, out_prefix, id_col=id_col)
        else:
            tcr_csv = str(tcr_csv)

        print("\n=== UNLABELED MODE: no cluster/leiden labels required ===")
        print(f"  TCR CSV: {tcr_csv}")

        tcr_csv, trd_seq_csv, trg_seq_csv = self._step1_from_csv(str(tcr_csv), out_prefix, id_col=id_col)

        results: Dict[str, object] = {
            "tcr_csv": tcr_csv,
            "trd_seq_csv": trd_seq_csv,
            "trg_seq_csv": trg_seq_csv,
            "esmc_unlabeled": None,
            "probert_unlabeled": None,
            "prediction_csv": None,
        }

        if run_esmc:
            if (esmc_trd_emb_path is None) ^ (esmc_trg_emb_path is None):
                raise ValueError("Provide both esmc_trd_emb_path and esmc_trg_emb_path, or neither.")
            trd_pt = str(esmc_trd_emb_path or self.output_root / f"{out_prefix}_trd_esmc_embeddings.pt")
            trg_pt = str(esmc_trg_emb_path or self.output_root / f"{out_prefix}_trg_esmc_embeddings.pt")
            if esmc_trd_emb_path is not None:
                print("\n=== UNLABELED STEP: Reuse existing ESMC embeddings ===")
                print(f"  TRD ESMC embeddings: {trd_pt}")
                print(f"  TRG ESMC embeddings: {trg_pt}")
                results["esmc_unlabeled"] = self._build_unlabeled_tensors(
                    tcr_csv=tcr_csv,
                    trd_emb_path=trd_pt,
                    trg_emb_path=trg_pt,
                    emb_type="esmc",
                    out_prefix=out_prefix,
                    id_col=id_col,
                    trd_seq_csv=trd_seq_csv,
                    trg_seq_csv=trg_seq_csv,
                )
            elif self._tcr is None:
                print("\n[SKIP] ESMC – pipeline_TCR not importable in this env.")
            else:
                print("\n=== UNLABELED STEP: ESMC embeddings ===")
                import pandas as pd
                import torch
                _trd_valid = (pd.read_csv(trd_seq_csv)["sequence"].astype(str).str.strip() != "None").sum()
                _trg_valid = (pd.read_csv(trg_seq_csv)["sequence"].astype(str).str.strip() != "None").sum()
                if _trd_valid == 0:
                    torch.save({}, trd_pt)
                    print(f"  [WARN] No valid TRD sequences -> {trd_pt}; TRD will be zero-filled.")
                else:
                    self._tcr.run_esm_embeddings(
                        seq_csv=trd_seq_csv,
                        model_weights=self.esmc_trd_weights,
                        out_file=trd_pt,
                        chunk_size=self.esmc_chunk_size,
                    )
                if _trg_valid == 0:
                    torch.save({}, trg_pt)
                    print(f"  [WARN] No valid TRG sequences -> {trg_pt}; TRG will be zero-filled.")
                else:
                    self._tcr.run_esm_embeddings(
                        seq_csv=trg_seq_csv,
                        model_weights=self.esmc_trg_weights,
                        out_file=trg_pt,
                        chunk_size=self.esmc_chunk_size,
                    )
                results["esmc_unlabeled"] = self._build_unlabeled_tensors(
                    tcr_csv=tcr_csv,
                    trd_emb_path=trd_pt,
                    trg_emb_path=trg_pt,
                    emb_type="esmc",
                    out_prefix=out_prefix,
                    id_col=id_col,
                    trd_seq_csv=trd_seq_csv,
                    trg_seq_csv=trg_seq_csv,
                )

        if run_probert:
            if (probert_trd_emb_path is None) ^ (probert_trg_emb_path is None):
                raise ValueError("Provide both probert_trd_emb_path and probert_trg_emb_path, or neither.")
            trd_pkl = str(probert_trd_emb_path or self.output_root / f"{out_prefix}_trd_probert.pkl")
            trg_pkl = str(probert_trg_emb_path or self.output_root / f"{out_prefix}_trg_probert.pkl")
            if probert_trd_emb_path is not None:
                print("\n=== UNLABELED STEP: Reuse existing ProBERT embeddings ===")
                print(f"  TRD ProBERT embeddings: {trd_pkl}")
                print(f"  TRG ProBERT embeddings: {trg_pkl}")
                results["probert_unlabeled"] = self._build_unlabeled_tensors(
                    tcr_csv=tcr_csv,
                    trd_emb_path=trd_pkl,
                    trg_emb_path=trg_pkl,
                    emb_type="probert",
                    out_prefix=out_prefix,
                    id_col=id_col,
                    trd_seq_csv=trd_seq_csv,
                    trg_seq_csv=trg_seq_csv,
                )
            elif not _HAS_PROBERT:
                print("\n[SKIP] ProBERT – pipeline_probert not importable in this env.")
            else:
                print("\n=== UNLABELED STEP: ProBERT embeddings ===")
                self._probert_embed(trd_seq_csv, "sequence", self.probert_trd_model, trd_pkl, "TRD")
                self._probert_embed(trg_seq_csv, "sequence", self.probert_trg_model, trg_pkl, "TRG")
                results["probert_unlabeled"] = self._build_unlabeled_tensors(
                    tcr_csv=tcr_csv,
                    trd_emb_path=trd_pkl,
                    trg_emb_path=trg_pkl,
                    emb_type="probert",
                    out_prefix=out_prefix,
                    id_col=id_col,
                    trd_seq_csv=trd_seq_csv,
                    trg_seq_csv=trg_seq_csv,
                )

        if predict_model_path is not None:
            results["prediction_csv"] = self.predict_from_concat(
                model_path=predict_model_path,
                out_prefix=out_prefix,
                emb_type=predict_emb_type,
                probert_concat=results.get("probert_unlabeled"),
                esmc_concat=results.get("esmc_unlabeled"),
                device=device,
                alpha=alpha,
                batch_size=prediction_batch_size,
            )

        print("\n=== Unlabeled embedding/prediction done ===")
        print(f"  All outputs saved to: {self.output_root}")
        return results

    def _unlabeled_csv_from_h5ad(
        self,
        h5ad_path: str | Path,
        out_prefix: str,
        id_col: Optional[str] = None,
    ) -> str:
        """Export required TCR columns from ``adata.obs`` without requiring labels."""
        if not _HAS_SCANPY:
            raise ImportError("scanpy is required to load .h5ad files.")

        adata = sc.read_h5ad(str(h5ad_path))
        obs = adata.obs.copy()
        required = ["trd_sequence", "TRDV", "TRDJ", "trg_sequence", "TRGV", "TRGJ"]
        missing = [c for c in required if c not in obs.columns]
        if missing:
            raise ValueError(
                "The h5ad obs table is missing required TCR columns: "
                f"{missing}. Available columns include: {list(obs.columns)[:50]}"
            )

        extra_cols = ["dataset"] if "dataset" in obs.columns else []
        out = obs[required + extra_cols].copy()
        if id_col is not None and id_col in obs.columns:
            out.insert(0, id_col, obs[id_col].astype(str).values)
        elif "cell_id" in obs.columns:
            out.insert(0, "cell_id", obs["cell_id"].astype(str).values)
        else:
            out.insert(0, "cell_id", obs.index.astype(str))

        out_csv = str(self.output_root / f"{out_prefix}_unlabeled_tcr.csv")
        out.to_csv(out_csv, index=False)
        print(f"  Exported unlabeled TCR CSV from h5ad: {out_csv}")
        return out_csv

    @staticmethod
    def _is_valid_seq(x: Any) -> bool:
        if x is None:
            return False
        s = str(x).strip()
        return s not in {"", "NA", "NaN", "nan", "None", "none", "null"}

    @staticmethod
    def _to_1d_tensor(x: Any):
        """Convert one embedding-like object to a 1D float tensor.

        Returns ``None`` for metadata/string/object arrays instead of raising.
        This prevents ProBERT/ESMC files that contain extra keys such as
        ``sequence`` or ``labels`` from being misread as numeric embeddings.
        """
        import numpy as np
        import torch

        if isinstance(x, torch.Tensor):
            t = x.detach().float().cpu()
        else:
            arr = np.asarray(x)
            if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
                return None
            try:
                t = torch.as_tensor(arr, dtype=torch.float32).detach().cpu()
            except Exception:
                return None
        if t.ndim == 0:
            t = t.reshape(1)
        elif t.ndim > 1:
            t = t.reshape(-1)
        return t

    @classmethod
    def _load_embedding_map(cls, path: str | Path) -> Dict[str, Any]:
        """
        Load sequence -> embedding mapping from ESMC .pt or ProBERT .pkl files.

        This parser is intentionally conservative: only numeric tensors/arrays are
        accepted as embeddings. String/object arrays are skipped, which fixes
        errors like:
            TypeError: can't convert np.ndarray of type numpy.str_
        """
        import pickle
        import torch

        path = str(path)
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as f:
                obj = pickle.load(f)
        else:
            try:
                obj = torch.load(path, map_location="cpu")
            except Exception:

                obj = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(obj, dict) and len(obj) == 0:

            return {}

        def pick_emb(d: Dict[str, Any]):
            for k in ["embedding", "emb", "cls", "cls_embedding", "mean_embedding", "x", "repr"]:
                if k in d:
                    t = cls._to_1d_tensor(d[k])
                    if t is not None:
                        return t
            for v in d.values():
                t = cls._to_1d_tensor(v)
                if t is not None:
                    return t
            return None

        emb_map: Dict[str, Any] = {}

        if isinstance(obj, dict):
            seq_keys = ["sequences", "sequence", "seqs", "ids", "names"]
            emb_keys = ["embeddings", "embedding", "embs", "X", "x", "repr", "cls"]
            seqs = None
            embs = None
            for sk in seq_keys:
                if sk in obj:
                    seqs = obj[sk]
                    break
            for ek in emb_keys:
                if ek in obj:
                    maybe = obj[ek]

                    if not (isinstance(maybe, list) and maybe and isinstance(maybe[0], dict)):
                        embs = maybe
                        break
            if seqs is not None and embs is not None:
                try:
                    import numpy as np
                    import torch
                    seq_list = list(seqs)
                    if isinstance(embs, torch.Tensor):
                        emb_arr = embs.detach().cpu()
                    else:
                        emb_arr = np.asarray(embs)
                    if len(seq_list) == len(emb_arr):
                        for seq, emb in zip(seq_list, emb_arr):
                            if cls._is_valid_seq(seq):
                                t = cls._to_1d_tensor(emb)
                                if t is not None:
                                    emb_map[str(seq).strip()] = t
                        if emb_map:
                            return emb_map
                except Exception:
                    emb_map = {}

        if isinstance(obj, dict):

            numeric_count = 0
            for k, v in obj.items():
                if not cls._is_valid_seq(k):
                    continue
                t = cls._to_1d_tensor(v)
                if t is not None:
                    emb_map[str(k).strip()] = t
                    numeric_count += 1
            if numeric_count > 0:
                return emb_map

            for rows_key in ["records", "data", "embeddings", "results"]:
                if rows_key in obj and isinstance(obj[rows_key], list):
                    obj = obj[rows_key]
                    break
            else:

                for _, d in obj.items():
                    if isinstance(d, dict):
                        seq = d.get("sequence", d.get("seq", d.get("cdr3")))
                        emb = pick_emb(d)
                        if cls._is_valid_seq(seq) and emb is not None:
                            emb_map[str(seq).strip()] = emb
                if emb_map:
                    return emb_map

        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    seq = item.get("sequence", item.get("seq", item.get("cdr3")))
                    emb = pick_emb(item)
                    if cls._is_valid_seq(seq) and emb is not None:
                        emb_map[str(seq).strip()] = emb
                elif isinstance(item, (tuple, list)) and len(item) >= 2:
                    seq, emb = item[0], item[1]
                    if cls._is_valid_seq(seq):
                        t = cls._to_1d_tensor(emb)
                        if t is not None:
                            emb_map[str(seq).strip()] = t

        if not emb_map:
            raise ValueError(
                f"Could not parse numeric embeddings from {path}. Expected sequence->embedding "
                "dict or list of dicts with sequence/embedding keys."
            )
        return emb_map

    @classmethod
    def _load_seq_embedding_map(cls, emb_path: str | Path) -> Dict[str, Any]:
        """
        Load a ``{sequence -> tensor}`` map from a ProBERT pkl or ESMC pt file.

        Both backends now save sequence-keyed dicts after ``_probert_embed``
        and ``run_esm_embeddings`` complete.  This is the single place that
        knows how to open either format.
        """
        import pickle
        import torch

        path = str(emb_path)
        if path.endswith((".pkl", ".pickle")):
            with open(path, "rb") as f:
                obj = pickle.load(f)
        else:
            try:
                obj = torch.load(path, map_location="cpu")
            except Exception:
                obj = torch.load(path, map_location="cpu", weights_only=False)

        if isinstance(obj, dict) and len(obj) == 0:
            return {}

        if isinstance(obj, dict):
            seq_map: Dict[str, Any] = {}
            for k, v in obj.items():
                t = cls._to_1d_tensor(v)
                if t is not None and cls._is_valid_seq(k):
                    seq_map[str(k).strip()] = t
            if seq_map:
                print(f"  Loaded {len(seq_map):,} sequence-keyed embeddings from {path}")
                return seq_map

        return cls._load_embedding_map(emb_path)

    def _build_unlabeled_tensors(
        self,
        tcr_csv: str | Path,
        trd_emb_path: str | Path,
        trg_emb_path: str | Path,
        emb_type: str,
        out_prefix: str,
        id_col: Optional[str] = None,
        trd_seq_csv: Optional[str] = None,   # kept for API compat, unused
        trg_seq_csv: Optional[str] = None,   # kept for API compat, unused
    ) -> Dict[str, object]:
        """Build unlabeled records, zero-filling chains without sequence embeddings."""
        import numpy as np
        import pandas as pd
        import torch

        df = pd.read_csv(tcr_csv)

        for c in ["trd_sequence", "TRDV", "TRDJ", "trg_sequence", "TRGV", "TRGJ"]:
            if c not in df.columns:
                df[c] = "None"
        if "dataset" not in df.columns:
            df["dataset"] = "all"

        if id_col is None:
            if "cell_id" in df.columns:
                id_col = "cell_id"
            elif "barcode" in df.columns:
                id_col = "barcode"
            else:
                id_col = "__row_id__"
                df[id_col] = [f"row_{i}" for i in range(len(df))]
        elif id_col not in df.columns:
            raise ValueError(f"id_col={id_col!r} not found in {tcr_csv}")

        trd_seq_map = self._load_seq_embedding_map(trd_emb_path)
        trg_seq_map = self._load_seq_embedding_map(trg_emb_path)

        n_trd = sum(1 for s in df["trd_sequence"].astype(str).str.strip() if s in trd_seq_map)
        n_trg = sum(1 for s in df["trg_sequence"].astype(str).str.strip() if s in trg_seq_map)
        print(f"  TRD: {n_trd:,} / {len(df):,} cells have an embedding")
        print(f"  TRG: {n_trg:,} / {len(df):,} cells have an embedding")

        if n_trd == 0 and n_trg == 0:
            raise ValueError(
                f"No TRD or TRG sequences in {tcr_csv} matched any entry in the "
                f"embedding files.  Check that trd_sequence / trg_sequence values "
                f"are the same CDR3 strings that were embedded."
            )

        _any_trd = next(iter(trd_seq_map.values()), None)
        _any_trg = next(iter(trg_seq_map.values()), None)
        trd_dim = _any_trd.shape[0] if _any_trd is not None else (_any_trg.shape[0] if _any_trg is not None else 1)
        trg_dim = _any_trg.shape[0] if _any_trg is not None else trd_dim
        if not trd_seq_map:
            print(f"  [WARN] No TRD embeddings at all; TRD will be zero (dim={trd_dim}).")
        if not trg_seq_map:
            print(f"  [WARN] No TRG embeddings at all; TRG will be zero (dim={trg_dim}).")

        records:     List[dict] = []
        X_rows:      list = []
        meta_rows:   list = []
        impute_rows: list = []

        for i, row in df.iterrows():
            cell_id = str(row[id_col])
            dataset = str(row.get("dataset", "all"))
            trd_seq = str(row.get("trd_sequence", "")).strip()
            trg_seq = str(row.get("trg_sequence", "")).strip()

            if trd_seq in trd_seq_map:
                trd_emb    = self._to_1d_tensor(trd_seq_map[trd_seq]).float().cpu()
                trd_source = "observed"
            else:
                trd_emb    = torch.zeros(trd_dim, dtype=torch.float32)
                trd_source = "missing"

            if trg_seq in trg_seq_map:
                trg_emb    = self._to_1d_tensor(trg_seq_map[trg_seq]).float().cpu()
                trg_source = "observed"
            else:
                trg_emb    = torch.zeros(trg_dim, dtype=torch.float32)
                trg_source = "missing"

            fused = torch.cat([trd_emb, trg_emb], dim=0).float().cpu()

            rec = {
                "sample_id":    cell_id,
                "row_index":    int(i),
                "dataset":      dataset,
                "trd_sequence": trd_seq,
                "trg_sequence": trg_seq,
                "TRDV":         str(row.get("TRDV", "None")),
                "TRDJ":         str(row.get("TRDJ", "None")),
                "TRGV":         str(row.get("TRGV", "None")),
                "TRGJ":         str(row.get("TRGJ", "None")),
                "trd_source":   trd_source,
                "trg_source":   trg_source,
                "trd_emb":      trd_emb,
                "trg_emb":      trg_emb,
                "embedding":    fused,
                "x":            fused,
            }
            records.append(rec)
            X_rows.append(fused.numpy())
            meta_rows.append({k: v for k, v in rec.items()
                               if k not in {"trd_emb", "trg_emb", "embedding", "x"}})
            impute_rows.append({
                "row_index":       int(i),
                "cell_id":         cell_id,
                "dataset":         dataset,
                "trd_sequence":    trd_seq,
                "trg_sequence":    trg_seq,
                "trd_source":      trd_source,
                "trg_source":      trg_source,
                "trd_was_missing": trd_source == "missing",
                "trg_was_missing": trg_source == "missing",
            })

        X         = np.vstack(X_rows).astype("float32")
        meta      = pd.DataFrame(meta_rows)
        impute_df = pd.DataFrame(impute_rows)

        concat_path = str(self.output_root / f"{out_prefix}_{emb_type}_unlabeled_concat.pt")
        x_path      = str(self.output_root / f"{out_prefix}_{emb_type}_unlabeled_X.npy")
        meta_path   = str(self.output_root / f"{out_prefix}_{emb_type}_unlabeled_metadata.csv")
        impute_path = str(self.output_root / f"{out_prefix}_{emb_type}_unlabeled_missing_chain_handling.csv")

        torch.save(records, concat_path)
        np.save(x_path, X)
        meta.to_csv(meta_path, index=False)
        impute_df.to_csv(impute_path, index=False)

        n_trd_miss = int(impute_df["trd_was_missing"].sum())
        n_trg_miss = int(impute_df["trg_was_missing"].sum())
        print(f"  Built {emb_type} unlabeled records: {len(records):,}")
        print(f"  X matrix: {X.shape}  ->  {x_path}")
        print(f"  TRD missing (zero-filled): {n_trd_miss:,} / {len(records):,}")
        print(f"  TRG missing (zero-filled): {n_trg_miss:,} / {len(records):,}")
        print(f"  Concat  ->  {concat_path}")
        print(f"  Metadata  ->  {meta_path}")
        print(f"  Missing-chain log  ->  {impute_path}")

        return {
            "records":                     records,
            "X":                           X,
            "metadata":                    meta,
            "concat_path":                 concat_path,
            "x_path":                      x_path,
            "metadata_path":               meta_path,
            "missing_path":                impute_path,
            "missing_chain_handling_path": impute_path,
        }

    def predict_unlabeled_from_built(
        self,
        built: Dict[str, object],
        model_path: str | Path,
        out_prefix: str = "newdata",
        emb_type: str = "probert",
        device: Optional[str] = None,
        alpha: float = 0.5,
    ) -> str:
        """Predict from an unlabeled result using the shared classifier path."""
        emb_type = _validate_prediction_mode(emb_type)
        probert = built.get("probert_unlabeled")
        esmc = built.get("esmc_unlabeled")
        if "records" in built or "concat_path" in built:
            if emb_type == "esmc":
                esmc = esmc if esmc is not None else built
            else:
                probert = probert if probert is not None else built
        pred_csv = self.predict_from_concat(
            model_path=model_path,
            out_prefix=out_prefix,
            emb_type=emb_type,
            probert_concat=probert,
            esmc_concat=esmc,
            device=device,
            alpha=alpha,
        )
        legacy_path = self.output_root / f"{out_prefix}_{emb_type}_predicted_clusters.csv"
        Path(pred_csv).replace(legacy_path)
        return str(legacy_path)

    def predict_from_existing_embeddings(
        self,
        model_path: str | Path,
        out_prefix: str = "newdata",
        emb_type: str = "esmc",
        tcr_csv: Optional[str | Path] = None,
        probert_concat_path: Optional[str | Path] = None,
        esmc_concat_path: Optional[str | Path] = None,
        probert_trd_emb_path: Optional[str | Path] = None,
        probert_trg_emb_path: Optional[str | Path] = None,
        esmc_trd_emb_path: Optional[str | Path] = None,
        esmc_trg_emb_path: Optional[str | Path] = None,
        cell_id_col: Optional[str] = None,
        sample_id_col: Optional[str] = None,
        dataset_col: Optional[str] = None,
        device: Optional[str] = None,
        alpha: Optional[float] = None,
        batch_size: int = 512,
    ) -> str:
        """Predict from concat files or sequence embeddings plus a metadata CSV."""
        emb_type = _validate_prediction_mode(emb_type)

        def _paired(name: str, trd_path, trg_path) -> bool:
            if (trd_path is None) ^ (trg_path is None):
                raise ValueError(f"Provide both {name}_trd_emb_path and {name}_trg_emb_path, or neither.")
            return trd_path is not None and trg_path is not None

        has_probert_pair = _paired("probert", probert_trd_emb_path, probert_trg_emb_path)
        has_esmc_pair = _paired("esmc", esmc_trd_emb_path, esmc_trg_emb_path)

        probert_concat: Optional[Any] = probert_concat_path
        esmc_concat: Optional[Any] = esmc_concat_path

        need_probert = emb_type in {"probert", "fused"}
        need_esmc = emb_type in {"esmc", "fused"}
        need_any_meta = emb_type == "vj"

        if probert_concat is None and (need_probert or (need_any_meta and esmc_concat is None)) and has_probert_pair:
            if tcr_csv is None:
                raise ValueError("tcr_csv is required when building ProBERT concat from existing sequence embeddings.")
            print("\n=== PREDICT ONLY: Build ProBERT concat from existing sequence embeddings ===")
            probert_concat = self._build_sequence_keyed_concat_tensors(
                tcr_csv=tcr_csv,
                trd_emb_path=probert_trd_emb_path,
                trg_emb_path=probert_trg_emb_path,
                emb_type="probert",
                out_prefix=out_prefix,
                cell_id_col=cell_id_col,
                sample_id_col=sample_id_col,
                dataset_col=dataset_col,
            )

        if esmc_concat is None and (need_esmc or (need_any_meta and probert_concat is None)) and has_esmc_pair:
            if tcr_csv is None:
                raise ValueError("tcr_csv is required when building ESMC concat from existing sequence embeddings.")
            print("\n=== PREDICT ONLY: Build ESMC concat from existing sequence embeddings ===")
            esmc_concat = self._build_sequence_keyed_concat_tensors(
                tcr_csv=tcr_csv,
                trd_emb_path=esmc_trd_emb_path,
                trg_emb_path=esmc_trg_emb_path,
                emb_type="esmc",
                out_prefix=out_prefix,
                cell_id_col=cell_id_col,
                sample_id_col=sample_id_col,
                dataset_col=dataset_col,
            )

        if need_probert and probert_concat is None:
            raise ValueError(
                f"emb_type={emb_type!r} requires --probert_concat_path or both "
                "--probert_trd_emb_path and --probert_trg_emb_path."
            )
        if need_esmc and esmc_concat is None:
            raise ValueError(
                f"emb_type={emb_type!r} requires --esmc_concat_path or both "
                "--esmc_trd_emb_path and --esmc_trg_emb_path."
            )
        if need_any_meta and probert_concat is None and esmc_concat is None:
            raise ValueError(
                "emb_type='vj' requires at least one concat path, or a TCR CSV plus "
                "one existing TRD/TRG embedding pair to build metadata records."
            )

        return self.predict_from_concat(
            model_path=model_path,
            out_prefix=out_prefix,
            emb_type=emb_type,
            probert_concat=probert_concat,
            esmc_concat=esmc_concat,
            device=device,
            alpha=alpha,
            batch_size=batch_size,
        )

    def predict_from_concat(
        self,
        model_path: str | Path,
        out_prefix: str = "newdata",
        emb_type: str = "fused",
        probert_concat: Optional[Any] = None,
        esmc_concat: Optional[Any] = None,
        device: Optional[str] = None,
        alpha: Optional[float] = None,
        batch_size: int = 512,
        vj_mode: Optional[str] = None,
    ) -> str:
        """Predict with esmc, probert, vj, or fused classifiers from records or concat paths."""
        emb_type = _validate_prediction_mode(emb_type)
        import json
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model_dir = Path(model_path)
        if not model_dir.is_dir():
            raise ValueError(f"model_path must be a model run directory, got: {model_dir}")

        hp_path = model_dir / "hparams.json"
        if not hp_path.exists():
            raise FileNotFoundError(f"hparams.json not found in {model_dir}")
        with open(hp_path) as f:
            hp = json.load(f)

        def _find_file(names):
            for name in names:
                q = model_dir / name
                if q.exists():
                    return q
                q = model_dir.parent / name
                if q.exists():
                    return q
            return None

        vocab_path = _find_file(["vj_vocab.json"])
        if vocab_path is None:
            raise FileNotFoundError(
                f"vj_vocab.json not found in {model_dir} or {model_dir.parent}."
            )
        with open(vocab_path) as f:
            vj_info = json.load(f)

        label_map_path = _find_file(["label_map.json"])
        label_map = None
        if label_map_path is not None:
            with open(label_map_path) as f:
                raw_label_map = json.load(f)
            label_map = {int(k): int(v) for k, v in raw_label_map.items()}
            rev_label_map = {v: k for k, v in label_map.items()}
        else:
            rev_label_map = None
            print("  [WARN] label_map.json not found; prediction labels will be model class indices.")

        num_classes = int(hp.get("num_classes", 0))
        if num_classes <= 0:
            if label_map:
                num_classes = len(set(label_map.values()))
            else:
                raise ValueError("num_classes missing from hparams.json and no label_map.json found.")

        chain_mode = hp.get("chain_mode", "both")
        if alpha is None:
            alpha = float(hp.get("alpha", 0.5))
        vj_mode = vj_mode or hp.get("vj_mode", "allele")

        def _load_records(obj, name):
            if obj is None:
                return None
            if isinstance(obj, (str, Path)):
                try:
                    return torch.load(str(obj), map_location="cpu")
                except Exception:
                    return torch.load(str(obj), map_location="cpu", weights_only=False)
            if isinstance(obj, dict):
                if "records" in obj:
                    return obj["records"]
                if "concat_path" in obj:
                    try:
                        return torch.load(str(obj["concat_path"]), map_location="cpu")
                    except Exception:
                        return torch.load(str(obj["concat_path"]), map_location="cpu", weights_only=False)
            if isinstance(obj, list):
                return obj
            raise TypeError(
                f"{name} must be a concat record list, a dict with records/concat_path, or a *_concat.pt path. "
                f"Got {type(obj).__name__}."
            )

        probert_records = _load_records(probert_concat, "probert_concat")
        esmc_records = _load_records(esmc_concat, "esmc_concat")

        need_probert = emb_type in {"probert", "fused"}
        need_esmc = emb_type in {"esmc", "fused"}
        need_any_meta = emb_type == "vj"

        if need_probert and probert_records is None:
            raise ValueError(f"emb_type={emb_type!r} requires probert_concat.")
        if need_esmc and esmc_records is None:
            raise ValueError(f"emb_type={emb_type!r} requires esmc_concat.")
        if need_any_meta and probert_records is None and esmc_records is None:
            raise ValueError("emb_type='vj' requires either probert_concat or esmc_concat for metadata.")

        def _cell_id(rec, i):
            return str(rec.get("cell_id", rec.get("sample_id", f"row_{i}")))

        def _align_records(primary, secondary=None):
            """Return records aligned by cell_id, preserving primary order."""
            if primary is None:
                primary = secondary
                secondary = None
            if secondary is None:
                return list(primary), None
            sec_map = {_cell_id(r, i): r for i, r in enumerate(secondary)}
            p_aligned, s_aligned, missing = [], [], []
            for i, r in enumerate(primary):
                cid = _cell_id(r, i)
                if cid in sec_map:
                    p_aligned.append(r)
                    s_aligned.append(sec_map[cid])
                else:
                    missing.append(cid)
            if missing:
                print(f"  [WARN] {len(missing):,} primary records were missing in the second concat and were skipped.")
            if not p_aligned:
                raise ValueError("No overlapping cell_id values between ProBERT and ESMC concat records.")
            return p_aligned, s_aligned

        if probert_records is not None and esmc_records is not None and emb_type == "fused":
            probert_records, esmc_records = _align_records(probert_records, esmc_records)
            meta_records = probert_records
        else:
            if emb_type == "esmc":
                meta_records = esmc_records
            elif emb_type == "probert":
                meta_records = probert_records
            else:
                meta_records = probert_records if probert_records is not None else esmc_records
            meta_records, _ = _align_records(meta_records, None)

        meta = pd.DataFrame([
            {k: v for k, v in rec.items()
             if k not in {"trd_emb", "trg_emb", "embedding", "x", "cls_last"}}
            for rec in meta_records
        ])

        def _tensor_1d(x):
            if isinstance(x, torch.Tensor):
                t = x.detach().cpu().float()
            else:
                t = torch.tensor(x, dtype=torch.float32).detach().cpu()
            if t.ndim == 0:
                t = t.reshape(1)
            elif t.ndim > 1:
                t = t.reshape(-1)
            return t

        def _stack_cdr3(records, label):
            rows = []
            for rec in records:
                if "trd_emb" in rec and "trg_emb" in rec:
                    x = torch.cat([_tensor_1d(rec["trd_emb"]), _tensor_1d(rec["trg_emb"])], dim=0)
                elif "cls_last" in rec:
                    x = _tensor_1d(rec["cls_last"])
                elif "embedding" in rec:
                    x = _tensor_1d(rec["embedding"])
                elif "x" in rec:
                    x = _tensor_1d(rec["x"])
                else:
                    raise KeyError(f"Cannot find embedding in {label} record. Expected trd_emb/trg_emb or cls_last/embedding/x.")
                rows.append(x)
            X = torch.stack(rows).float()
            print(f"  {label} concat tensor: {tuple(X.shape)}  per-chain dim={X.shape[1] // 2}")
            return X

        x_probert = _stack_cdr3(probert_records, "ProBERT") if need_probert else None
        x_esmc = _stack_cdr3(esmc_records, "ESMC") if need_esmc else None

        trd_mask = torch.tensor(vj_info["trd_indices"], dtype=torch.long)
        trg_mask = torch.tensor(vj_info["trg_indices"], dtype=torch.long)

        def make_mlp(input_dim, hidden_dim, depth, dropout=0.1):
            layers = []
            dim = input_dim
            for _ in range(depth):
                layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
                dim = hidden_dim
            return nn.Sequential(*layers)

        class JointCDR3VJClassifier(nn.Module):
            def __init__(self, in_dim_cdr3, vj_input_dim, d_hidden, g_hidden, vj_hidden,
                         d_depth, g_depth, vj_depth, num_classes, dropout=0.1,
                         chain_mode="both", trd_mask=None, trg_mask=None):
                super().__init__()
                self.half_dim = in_dim_cdr3 // 2
                self.chain_mode = chain_mode
                self.vj_tower = make_mlp(vj_input_dim, vj_hidden, vj_depth, dropout)
                self.d_tower = make_mlp(self.half_dim, d_hidden, d_depth, dropout)
                self.g_tower = make_mlp(self.half_dim, g_hidden, g_depth, dropout)
                self.head_joint = nn.Linear(d_hidden + g_hidden + vj_hidden, num_classes)
                self.register_buffer("trd_mask", trd_mask)
                self.register_buffer("trg_mask", trg_mask)
            def forward(self, cdr3_emb, vj_vec):
                d_emb = cdr3_emb[:, :self.half_dim]
                g_emb = cdr3_emb[:, self.half_dim:]
                curr_vj = vj_vec.clone()
                if self.chain_mode == "trd":
                    g_emb = torch.zeros_like(g_emb)
                    if self.trg_mask is not None:
                        curr_vj[:, self.trg_mask] = 0.0
                elif self.chain_mode == "trg":
                    d_emb = torch.zeros_like(d_emb)
                    if self.trd_mask is not None:
                        curr_vj[:, self.trd_mask] = 0.0
                return self.head_joint(torch.cat([self.d_tower(d_emb), self.g_tower(g_emb), self.vj_tower(curr_vj)], dim=1))

        class VJOnlyClassifier(nn.Module):
            def __init__(self, vj_input_dim, vj_hidden, vj_depth, num_classes, dropout=0.1,
                         chain_mode="both", trd_mask=None, trg_mask=None):
                super().__init__()
                self.vj_tower = make_mlp(vj_input_dim, vj_hidden, vj_depth, dropout)
                self.head = nn.Linear(vj_hidden, num_classes)
                self.chain_mode = chain_mode
                self.register_buffer("trd_mask", trd_mask)
                self.register_buffer("trg_mask", trg_mask)
            def forward(self, vj_vec):
                curr_vj = vj_vec.clone()
                if self.chain_mode == "trd":
                    if self.trg_mask is not None:
                        curr_vj[:, self.trg_mask] = 0.0
                elif self.chain_mode == "trg":
                    if self.trd_mask is not None:
                        curr_vj[:, self.trd_mask] = 0.0
                return self.head(self.vj_tower(curr_vj))

        def _safe_torch_load(path):
            try:
                obj = torch.load(str(path), map_location=device)
            except Exception:
                obj = torch.load(str(path), map_location=device, weights_only=False)
            if isinstance(obj, dict) and "state_dict" in obj:
                obj = obj["state_dict"]
            return obj

        def _ckpt_half_dim(sd):
            for key in ("d_tower.0.weight", "g_tower.0.weight"):
                if key in sd:
                    return int(sd[key].shape[1])
            return None

        def _load_compatible_ckpt(filename, x_dim, role):
            path = model_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"{filename} not found in {model_dir}")
            sd = _safe_torch_load(path)
            half_dim = _ckpt_half_dim(sd)
            if x_dim % 2 or half_dim != x_dim // 2:
                raise ValueError(
                    f"{role}: checkpoint per-chain dimension {half_dim} "
                    f"does not match concat dimension {x_dim}."
                )
            return sd

        def _build_joint(sd, in_dim):
            model = JointCDR3VJClassifier(
                in_dim, vj_info["total_dim"],
                hp["D_HIDDEN"], hp["G_HIDDEN"], hp["VJ_HIDDEN"],
                hp["D_DEPTH"], hp["G_DEPTH"], hp["VJ_DEPTH"],
                num_classes, hp["DROPOUT"], chain_mode, trd_mask, trg_mask,
            ).to(device)
            model.load_state_dict(sd, strict=False)
            return model

        def _strip_vj(s):
            return str(s).split("*")[0].strip() if vj_mode == "gene" else str(s).strip()

        def _build_vj_matrix(meta_df):
            bad = {"nan", "none", "na", "null", "", "<na>"}
            vecs = []
            vocabs = vj_info["vocabs"]
            offsets = vj_info["offsets"]
            col_map = {
                "TRDV": "TRDV", "TRDV_gene": "TRDV",
                "TRDJ": "TRDJ", "TRDJ_gene": "TRDJ",
                "TRGV": "TRGV", "TRGV_gene": "TRGV",
                "TRGJ": "TRGJ", "TRGJ_gene": "TRGJ",
            }
            for _, row in meta_df.iterrows():
                vec = torch.zeros(vj_info["total_dim"], dtype=torch.float32)
                for part_i, (_id_col, pair) in enumerate(vocabs.items()):
                    vocab, src_col = pair[0], pair[1]
                    meta_col = col_map.get(src_col, src_col)
                    raw = row.get(meta_col, "")
                    val = _strip_vj(raw)
                    if val.lower() in bad:
                        continue
                    tok_id = vocab.get(val, -1)
                    if tok_id >= 0:
                        vec[offsets[part_i] + int(tok_id)] = 1.0
                vecs.append(vec)
            if not vecs:
                return torch.zeros((0, vj_info["total_dim"]), dtype=torch.float32)
            return torch.stack(vecs).float()

        vj = _build_vj_matrix(meta)
        print(f"  VJ matrix: {tuple(vj.shape)}  mode={vj_mode}  chain_mode={chain_mode}")

        @torch.no_grad()
        def _infer(model, *inputs):
            model.eval()
            n = int(inputs[0].shape[0])
            outs = []
            for start in range(0, n, batch_size):
                sl = slice(start, start + batch_size)
                outs.append(model(*[x[sl].to(device) for x in inputs]).detach().cpu())
            return torch.cat(outs, dim=0)

        logits_parts = []
        weights = []

        if emb_type == "esmc":
            sd = _load_compatible_ckpt("best_model_esmc.pt", x_esmc.shape[1], "ESMC joint")
            logits_parts.append(_infer(_build_joint(sd, x_esmc.shape[1]), x_esmc, vj)); weights.append(1.0)
        elif emb_type == "probert":
            sd = _load_compatible_ckpt("best_model_probert.pt", x_probert.shape[1], "ProBERT joint")
            logits_parts.append(_infer(_build_joint(sd, x_probert.shape[1]), x_probert, vj)); weights.append(1.0)
        elif emb_type == "vj":
            vj_path = model_dir / "best_model_vj.pt"
            if not vj_path.exists():
                raise FileNotFoundError(f"best_model_vj.pt not found in {model_dir}")
            sd = _safe_torch_load(vj_path)
            model = VJOnlyClassifier(
                vj_info["total_dim"], hp["VJ_HIDDEN"], hp["VJ_DEPTH"],
                num_classes, hp["DROPOUT"], chain_mode, trd_mask, trg_mask,
            ).to(device)
            model.load_state_dict(sd, strict=False)
            logits_parts.append(_infer(model, vj)); weights.append(1.0)
        elif emb_type == "fused":
            sd_p = _load_compatible_ckpt("best_model_probert.pt", x_probert.shape[1], "ProBERT joint")
            sd_e = _load_compatible_ckpt("best_model_esmc.pt", x_esmc.shape[1], "ESMC joint")
            logits_parts.append(_infer(_build_joint(sd_p, x_probert.shape[1]), x_probert, vj)); weights.append(1.0 - float(alpha))
            logits_parts.append(_infer(_build_joint(sd_e, x_esmc.shape[1]), x_esmc, vj)); weights.append(float(alpha))
        total_w = float(sum(weights)) if sum(weights) != 0 else 1.0
        logits = sum(w * l for w, l in zip(weights, logits_parts)) / total_w
        probs = torch.softmax(logits, dim=1).numpy()
        pred_idx = probs.argmax(axis=1)
        pred_label = [rev_label_map.get(int(i), int(i)) for i in pred_idx] if rev_label_map is not None else [int(i) for i in pred_idx]

        out = meta.copy()
        out["pred_cluster_idx"] = pred_idx
        out["pred_cluster_label"] = pred_label
        out["pred_confidence"] = probs.max(axis=1)
        for j in range(probs.shape[1]):
            label = rev_label_map.get(j, j) if rev_label_map is not None else j
            out[f"prob_cluster_{label}"] = probs[:, j]

        pred_csv = str(self.output_root / f"{out_prefix}_{emb_type}_predictions.csv")
        out.to_csv(pred_csv, index=False)
        print(f"  Predictions ({len(out):,} rows) -> {pred_csv}")
        return pred_csv

    @staticmethod
    def print_concat_summary(concat: Optional[List[dict]], max_rows: int = 3) -> None:
        """Print record counts, tensor shapes, and a few metadata rows."""
        if not concat:
            print("No concat records available.")
            return

        import torch

        print(f"Concat records: {len(concat):,}")
        for key, value in concat[0].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        for record in concat[:max_rows]:
            print({key: value for key, value in record.items()
                   if not isinstance(value, torch.Tensor)})


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build ESMC and/or ProBERT TCR embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--h5ad_path", type=str, help="Path to AnnData .h5ad file.")
    src.add_argument("--tcr_csv",   type=str, help="Path to existing per-cell TCR CSV.")

    p.add_argument(
        "--leiden_cols",
        nargs="+",
        default=None,
        metavar="COL",
        help=(
            "One or more leiden cluster column names to carry through the "
            "pipeline and store in every tensor record. "
            "Example: --leiden_cols leiden_0.2 leiden_0.4 leiden_0.8  "
            "(default: leiden_0.4)"
        ),
    )
    p.add_argument(
        "--cluster_col",
        default=None,
        help=(
            "Primary cluster column used for keep_clusters / merge_rules "
            "filtering. Defaults to the first entry of --leiden_cols."
        ),
    )
    p.add_argument("--batch_col",     default="batch")
    p.add_argument("--target_batch",  default="batch01")
    p.add_argument("--keep_clusters", nargs="+", default=["1","2","3","4","5","6"])

    p.add_argument("--output_root", required=True)
    p.add_argument("--out_prefix",  default="cohort3")

    p.add_argument("--esmc_trd_weights",  default="")
    p.add_argument("--esmc_trg_weights",  default="")
    p.add_argument("--probert_trd_model", default="")
    p.add_argument("--probert_trg_model", default="")
    p.add_argument("--probert_base_path", default="../models/proteinbert",
                   help="Base pretrained ProteinBERT weights dir (used by from_pretrained).")

    p.add_argument("--unlabeled", action="store_true", help="Build embeddings for new data without cluster/leiden labels.")
    p.add_argument("--predict_only", action="store_true", help="Skip embedding inference and predict from existing concat or sequence embedding files.")
    p.add_argument("--predict_model_path", default=None, help="Optional classifier model for unlabeled prediction.")
    p.add_argument(
        "--predict_emb_type",
        default="esmc",
        choices=_PREDICTION_MODES,
        help="Embedding branch/checkpoint to use for prediction.",
    )
    p.add_argument("--probert_concat_path", default=None, help="Existing ProBERT cell-level concat .pt path for prediction-only mode.")
    p.add_argument("--esmc_concat_path", default=None, help="Existing ESMC cell-level concat .pt path for prediction-only mode.")
    p.add_argument("--probert_trd_emb_path", default=None, help="Existing sequence-keyed TRD ProBERT embedding .pkl path.")
    p.add_argument("--probert_trg_emb_path", default=None, help="Existing sequence-keyed TRG ProBERT embedding .pkl path.")
    p.add_argument("--esmc_trd_emb_path", default=None, help="Existing sequence-keyed TRD ESMC embedding .pt path.")
    p.add_argument("--esmc_trg_emb_path", default=None, help="Existing sequence-keyed TRG ESMC embedding .pt path.")
    p.add_argument("--prediction_alpha", type=float, default=None, help="ESMC weight for fused prediction. Defaults to hparams alpha or 0.5.")
    p.add_argument("--prediction_batch_size", type=int, default=512, help="Batch size for classifier inference.")
    p.add_argument("--device", default=None, help="Prediction device, e.g. cuda, cuda:0, or cpu. Defaults to cuda when available.")
    p.add_argument("--cell_id_col", default=None, help="Input column to copy into cell_id. This should uniquely identify each row/cell.")
    p.add_argument("--sample_id_col", default=None, help="Input column to copy into sample_id. This is sample/group metadata used for missing-chain summaries, e.g. sample_id or dataset.")
    p.add_argument("--dataset_col", default=None, help="Input column to copy into dataset. Defaults to dataset if present, otherwise sample_id.")
    p.add_argument("--id_col", default=None, help="Backward-compatible alias for --cell_id_col.")
    p.add_argument("--reference", action="store_true", help="Normalize non-NA TRDV/TRGV/TRDJ/TRGJ to allele format; add *01 when no allele is present, and save computed metadata CSV.")

    p.add_argument("--run_esmc",    dest="run_esmc",    action="store_true",  default=True)
    p.add_argument("--no_esmc",     dest="run_esmc",    action="store_false")
    p.add_argument("--run_probert", dest="run_probert", action="store_true",  default=True)
    p.add_argument("--no_probert",  dest="run_probert", action="store_false")

    p.add_argument("--esmc_chunk_size",    type=int, default=256)
    p.add_argument("--probert_batch_size", type=int, default=64)
    p.add_argument("--probert_max_length", type=int, default=45)
    return p

def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if not args.predict_only and args.h5ad_path is None and args.tcr_csv is None:
        parser.error("one of --h5ad_path or --tcr_csv is required unless --predict_only is used.")
    if args.predict_only and args.predict_model_path is None:
        parser.error("--predict_only requires --predict_model_path.")

    pipe = EmbeddingPipeline(
        output_root        = args.output_root,
        esmc_trd_weights   = args.esmc_trd_weights,
        esmc_trg_weights   = args.esmc_trg_weights,
        probert_trd_model  = args.probert_trd_model,
        probert_trg_model  = args.probert_trg_model,
        probert_base_path  = args.probert_base_path,
        leiden_cols        = args.leiden_cols,
        esmc_chunk_size    = args.esmc_chunk_size,
        probert_batch_size = args.probert_batch_size,
        probert_max_length = args.probert_max_length,
    )

    if args.predict_only:
        return pipe.predict_from_existing_embeddings(
            model_path           = args.predict_model_path,
            out_prefix           = args.out_prefix,
            emb_type             = args.predict_emb_type,
            tcr_csv              = args.tcr_csv,
            probert_concat_path  = args.probert_concat_path,
            esmc_concat_path     = args.esmc_concat_path,
            probert_trd_emb_path = args.probert_trd_emb_path,
            probert_trg_emb_path = args.probert_trg_emb_path,
            esmc_trd_emb_path    = args.esmc_trd_emb_path,
            esmc_trg_emb_path    = args.esmc_trg_emb_path,
            cell_id_col          = args.cell_id_col if args.cell_id_col is not None else args.id_col,
            sample_id_col        = args.sample_id_col,
            dataset_col          = args.dataset_col,
            device               = args.device,
            alpha                = args.prediction_alpha,
            batch_size           = args.prediction_batch_size,
        )

    if args.unlabeled:
        results = pipe.run_unlabeled(
            h5ad_path          = args.h5ad_path,
            tcr_csv            = args.tcr_csv,
            out_prefix         = args.out_prefix,
            run_esmc           = args.run_esmc,
            run_probert        = args.run_probert,
            esmc_trd_emb_path  = args.esmc_trd_emb_path,
            esmc_trg_emb_path  = args.esmc_trg_emb_path,
            probert_trd_emb_path = args.probert_trd_emb_path,
            probert_trg_emb_path = args.probert_trg_emb_path,
            predict_model_path = args.predict_model_path,
            predict_emb_type   = args.predict_emb_type,
            id_col             = args.cell_id_col if args.cell_id_col is not None else args.id_col,
            device             = args.device,
            alpha              = args.prediction_alpha,
            prediction_batch_size = args.prediction_batch_size,
        )
        if results.get("esmc_unlabeled") is not None:
            EmbeddingPipeline.print_concat_summary(results["esmc_unlabeled"]["records"])
        if results.get("probert_unlabeled") is not None:
            EmbeddingPipeline.print_concat_summary(results["probert_unlabeled"]["records"])
        return results

    results = pipe.run(
        h5ad_path     = args.h5ad_path,
        tcr_csv       = args.tcr_csv,
        cluster_col   = args.cluster_col,
        batch_col     = args.batch_col,
        target_batch  = args.target_batch,
        keep_clusters = args.keep_clusters,
        merge_rules   = [],
        out_prefix    = args.out_prefix,
        run_esmc      = args.run_esmc,
        run_probert   = args.run_probert,
        cell_id_col   = args.cell_id_col if args.cell_id_col is not None else args.id_col,
        sample_id_col = args.sample_id_col,
        dataset_col   = args.dataset_col,
        reference     = args.reference,

    )

    EmbeddingPipeline.print_concat_summary(results["esmc_concat"])
    EmbeddingPipeline.print_concat_summary(results["probert_concat"])
    return results

if __name__ == "__main__":
    main()
