# =============================================================================
# pipeline_embedding.py
#
# Callable from a Python script OR a Jupyter notebook:
#
#   from gdtcr.pipeline_embedding import EmbeddingPipeline
#
#   pipe = EmbeddingPipeline(
#       output_root        = "/path/to/output/fig3_embeddings",
#       esmc_trd_weights   = "/path/to/esmc_trd.pt",
#       esmc_trg_weights   = "/path/to/esmc_trg.pt",
#       probert_trd_model  = "/path/to/probert_trd.pth",
#       probert_trg_model  = "/path/to/probert_trg.pth",
#       probert_base_path  = "/path/to/models/proteinbert",
#       leiden_cols        = ["leiden_0.2", "leiden_0.4", "leiden_0.8"],  # ← NEW
#   )
#
#   # ── Entry point A: start from h5ad ──────────────────────────────────────
#   results = pipe.run(
#       h5ad_path     = "/path/to/adata.h5ad",
#       cluster_col   = "leiden_0.4",          # drives filtering / merging
#       leiden_cols   = ["leiden_0.2", "leiden_0.4", "leiden_0.8"],  # ← NEW (optional override)
#       batch_col     = "batch",
#       target_batch  = "batch01",
#       keep_clusters = ["1","2","3","4","5","6"],
#       merge_rules   = [],
#       out_prefix    = "cohort3",
#   )
#
#   # ── Entry point B: start from an existing TCR CSV ───────────────────────
#   results = pipe.run(
#       tcr_csv    = "/path/to/cohort3_final_tcr_merged.csv",
#       out_prefix = "cohort3",
#   )
#
#   # ── Run only one model ───────────────────────────────────────────────────
#   results = pipe.run(..., run_esmc=True, run_probert=False)
#
#   # ── Inspect what is inside a concat result ───────────────────────────────
#   EmbeddingPipeline.print_concat_summary(results["esmc_concat"])
#
# Dependencies
#   pipeline_TCR.py     – TCRPipeline  (ESMC embeddings + tensor building)
#   pipeline_probert.py – run_eval     (ProBERT CLS embeddings)
#
# Environment notes
#   ESMC steps    → esmc conda env    (has `esm` installed)
#   ProBERT steps → probert conda env (has `tape` installed)
#   In the wrong env the unavailable model is skipped with a clear warning;
#   re-run in the correct env to fill in the missing pkl / pt files.
# =============================================================================

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default leiden columns used when none are specified anywhere
_DEFAULT_LEIDEN_COLS: List[str] = ["leiden_0.4"]

def _patch_sympy_printing_compat() -> None:
    """
    Patch SymPy so older TAPE/ProteinBERT dependency stacks can access
    ``sympy.printing.StrPrinter``.

    Some SymPy versions allow ``import sympy.printing`` but do not expose
    ``printing`` as an attribute on the top-level ``sympy`` module until the
    submodule is imported.  Older code with annotations such as
    ``sympy.printing.StrPrinter`` can therefore fail during import with:

        AttributeError: module 'sympy' has no attribute 'printing'

    This is safe to run even when SymPy is absent; in that case ProBERT will
    simply be skipped by the optional import block below.
    """
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


# Apply before importing pipeline_probert.
_patch_sympy_printing_compat()

# ---------------------------------------------------------------------------
# Optional heavy imports – resolved lazily
# ---------------------------------------------------------------------------
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


# =============================================================================
# EmbeddingPipeline
# =============================================================================
class EmbeddingPipeline:
    """
    End-to-end TCR embedding builder for ESMC and ProBERT models.

    Parameters
    ----------
    output_root : str | Path
        Single directory where ALL outputs are saved flat – processed CSVs,
        raw embedding .pt / .pkl files, and final per-cell tensor .pt files.
    esmc_trd_weights : str | Path
        Fine-tuned ESMC model weights for the TRD (delta) chain.
    esmc_trg_weights : str | Path
        Fine-tuned ESMC model weights for the TRG (gamma) chain.
    probert_trd_model : str | Path
        Fine-tuned ProBERT state-dict (.pth) for TRD.
    probert_trg_model : str | Path
        Fine-tuned ProBERT state-dict (.pth) for TRG.
    probert_base_path : str | Path
        Directory of the base pretrained ProteinBERT weights.
    leiden_cols : list of str, optional
        Leiden cluster column(s) to preserve in all CSV and tensor outputs.
        This acts as the pipeline-wide default; individual ``run()`` calls
        can override it.  Defaults to ["leiden_0.4"].
    esmc_chunk_size : int
        Sequences per GPU batch for ESMC inference (default 256).
    probert_batch_size : int
        DataLoader batch size for ProBERT inference (default 64).
    probert_max_length : int
        Token-length cap for ProBERT (default 45).
    """

    def __init__(
        self,
        output_root:        str | Path,
        esmc_trd_weights:   str | Path = "",
        esmc_trg_weights:   str | Path = "",
        probert_trd_model:  str | Path = "",
        probert_trg_model:  str | Path = "",
        probert_base_path:  str | Path = "../models/proteinbert",
        leiden_cols:        Optional[List[str]] = None,   # ← NEW
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

        # Pipeline-wide leiden column default
        self.leiden_cols: List[str] = leiden_cols if leiden_cols is not None else _DEFAULT_LEIDEN_COLS

        self.esmc_chunk_size    = esmc_chunk_size
        self.probert_batch_size = probert_batch_size
        self.probert_max_length = probert_max_length

        # TCRPipeline is initialised with the same leiden_cols so its own
        # internal default is consistent with ours.
        self._tcr = (
            TCRPipeline(base_outdir=str(self.output_root), leiden_cols=self.leiden_cols)
            if _HAS_TCR else None
        )

        self.output_root.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Public entry point
    # =========================================================================
    def run(
        self,
        # ── Entry point A: start from h5ad ──────────────────────────────────
        h5ad_path:     Optional[str | Path] = None,
        cluster_col:   Optional[str] = None,
        batch_col:     str = "batch",
        target_batch:  str = "batch01",
        keep_clusters: Optional[List[str]] = None,
        merge_rules:   Optional[List[Tuple]] = None,
        # ── Entry point B: start from an existing per-cell TCR CSV ──────────
        tcr_csv: Optional[str | Path] = None,
        # ── Shared ──────────────────────────────────────────────────────────
        out_prefix:  str  = "cohort3",
        run_esmc:    bool = True,
        run_probert: bool = True,
        # ── Leiden columns (overrides instance default for this call) ────────
        leiden_cols: Optional[List[str]] = None,          # ← NEW
    ) -> Dict[str, object]:
        """
        Run the full embedding pipeline.

        Provide **either** ``h5ad_path`` (build TCR table from scratch) **or**
        ``tcr_csv`` (skip directly to embedding). Exactly one is required.

        Parameters
        ----------
        leiden_cols : list of str, optional
            Leiden column(s) to carry into the exported CSV and all tensor
            records.  Overrides ``self.leiden_cols`` for this call only.
            Defaults to ``self.leiden_cols`` when not given.
        cluster_col : str, optional
            Primary cluster column that drives keep_clusters / merge_rules
            filtering.  Defaults to the first entry of the resolved
            ``leiden_cols`` when not provided.

        Returns
        -------
        dict with keys:
            tcr_csv        – path to the processed per-cell TCR CSV
            trd_seq_csv    – path to the unique TRD sequence CSV
            trg_seq_csv    – path to the unique TRG sequence CSV
            esmc_concat    – list of per-cell dicts from build_tensors, or None
            probert_concat – list of per-cell dicts from build_tensors, or None
        """
        if h5ad_path is None and tcr_csv is None:
            raise ValueError("Provide either h5ad_path or tcr_csv.")
        if h5ad_path is not None and tcr_csv is not None:
            raise ValueError("Provide h5ad_path OR tcr_csv, not both.")

        # Resolve leiden columns for this run
        active_leiden_cols: List[str] = leiden_cols if leiden_cols is not None else self.leiden_cols

        # cluster_col drives filtering; default to first leiden col
        if cluster_col is None:
            cluster_col = active_leiden_cols[0]

        print(f"\n  leiden_cols : {active_leiden_cols}")
        print(f"  cluster_col : {cluster_col}")

        results: Dict[str, object] = {
            "tcr_csv": None, "trd_seq_csv": None, "trg_seq_csv": None,
            "esmc_concat": None, "probert_concat": None,
        }

        # ── Step 1 ───────────────────────────────────────────────────────────
        if h5ad_path is not None:
            tcr_csv, trd_seq_csv, trg_seq_csv = self._step1_from_h5ad(
                h5ad_path, cluster_col, batch_col, target_batch,
                keep_clusters or [], merge_rules or [], out_prefix,
                active_leiden_cols,                        # ← passed through
            )
        else:
            tcr_csv, trd_seq_csv, trg_seq_csv = self._step1_from_csv(
                str(tcr_csv), out_prefix,
            )

        results.update(
            tcr_csv=tcr_csv, trd_seq_csv=trd_seq_csv, trg_seq_csv=trg_seq_csv
        )

        # ── Steps 2-3: ESMC ──────────────────────────────────────────────────
        if run_esmc:
            results["esmc_concat"] = self._run_esmc(
                tcr_csv, trd_seq_csv, trg_seq_csv, out_prefix,
                active_leiden_cols,                        # ← passed through
            )

        # ── Steps 4-5: ProBERT ───────────────────────────────────────────────
        if run_probert:
            results["probert_concat"] = self._run_probert(
                tcr_csv, trd_seq_csv, trg_seq_csv, out_prefix,
                active_leiden_cols,                        # ← passed through
            )

        print("\n=== All done ===")
        print(f"  All outputs saved to: {self.output_root}")
        return results

    # =========================================================================
    # Step 1 – build TCR table
    # =========================================================================
    def _step1_from_h5ad(
        self,
        h5ad_path, cluster_col, batch_col, target_batch,
        keep_clusters, merge_rules, out_prefix,
        leiden_cols: List[str],                            # ← NEW param
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
            leiden_cols   = leiden_cols,                   # ← forwarded
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

    def _step1_from_csv(self, tcr_csv: str, out_prefix: str) -> Tuple[str, str, str]:
        """
        Derive unique TRD / TRG sequence CSVs from an existing per-cell TCR CSV.

        The input CSV must have a header row and at minimum these columns:
            trd_sequence, TRDV, TRDJ, trg_sequence, TRGV, TRGJ
        All three output CSVs keep the header so ProBERT can consume them
        directly via seq_col="sequence".
        """
        print("\n=== STEP 1: Derive unique sequences from existing TCR CSV ===")
        import pandas as pd

        df = pd.read_csv(tcr_csv)
        _bad = {"NA", "nan", "None", "NaN"}

        def _extract_unique(df, v_col, seq_col_in, j_col, seq_col_out):
            for c in [v_col, seq_col_in, j_col]:
                if c not in df.columns:
                    df[c] = "None"
            out = (
                df[[v_col, seq_col_in, j_col]]
                .rename(columns={seq_col_in: "sequence"})
                .dropna(subset=["sequence"])
            )
            out = out[
                ~out.apply(lambda r: any(str(v) in _bad for v in r), axis=1)
            ].drop_duplicates(subset=["sequence"])
            return out

        df_trd = _extract_unique(df, "TRDV", "trd_sequence", "TRDJ", "sequence")
        df_trg = _extract_unique(df, "TRGV", "trg_sequence", "TRGJ", "sequence")

        trd_seq_csv = str(self.output_root / f"{out_prefix}_trd_unique.csv")
        trg_seq_csv = str(self.output_root / f"{out_prefix}_trg_unique.csv")
        df_trd.to_csv(trd_seq_csv, index=False)
        df_trg.to_csv(trg_seq_csv, index=False)

        print(f"  TCR CSV     : {tcr_csv}")
        print(f"  TRD seq CSV : {trd_seq_csv}  ({len(df_trd)} unique sequences)")
        print(f"  TRG seq CSV : {trg_seq_csv}  ({len(df_trg)} unique sequences)")
        return tcr_csv, trd_seq_csv, trg_seq_csv

    # =========================================================================
    # Steps 2-3 – ESMC
    # =========================================================================
    def _run_esmc(
        self,
        tcr_csv: str,
        trd_seq_csv: str,
        trg_seq_csv: str,
        out_prefix: str,
        leiden_cols: List[str],                            # ← NEW param
    ):
        if self._tcr is None:
            print("\n[SKIP] ESMC – pipeline_TCR not importable in this env.")
            return None

        print("\n=== STEP 2: ESMC embeddings ===")
        trd_pt = str(self.output_root / f"{out_prefix}_trd_esmc_embeddings.pt")
        trg_pt = str(self.output_root / f"{out_prefix}_trg_esmc_embeddings.pt")

        self._tcr.run_esm_embeddings(
            seq_csv=trd_seq_csv, model_weights=self.esmc_trd_weights,
            out_file=trd_pt, chunk_size=self.esmc_chunk_size,
        )
        self._tcr.run_esm_embeddings(
            seq_csv=trg_seq_csv, model_weights=self.esmc_trg_weights,
            out_file=trg_pt, chunk_size=self.esmc_chunk_size,
        )

        print("\n=== STEP 3: Build ESMC tensors ===")
        concat = self._tcr.build_tensors(
            tcr_csv      = tcr_csv,
            trd_emb_path = trd_pt,
            trg_emb_path = trg_pt,
            emb_type     = "esmc",
            out_prefix   = out_prefix,
            leiden_cols  = leiden_cols,                    # ← forwarded
        )
        print(f"  ESMC tensors → {self.output_root}")
        return concat

    # =========================================================================
    # Steps 4-5 – ProBERT
    # =========================================================================
    def _run_probert(
        self,
        tcr_csv: str,
        trd_seq_csv: str,
        trg_seq_csv: str,
        out_prefix: str,
        leiden_cols: List[str],                            # ← NEW param
    ):
        if not _HAS_PROBERT:
            print("\n[SKIP] ProBERT – pipeline_probert not importable in this env.")
            return None

        print("\n=== STEP 4: ProBERT embeddings ===")
        trd_pkl = str(self.output_root / f"{out_prefix}_trd_probert.pkl")
        trg_pkl = str(self.output_root / f"{out_prefix}_trg_probert.pkl")

        self._probert_embed(trd_seq_csv, "sequence", self.probert_trd_model, trd_pkl, "TRD")
        self._probert_embed(trg_seq_csv, "sequence", self.probert_trg_model, trg_pkl, "TRG")

        print("\n=== STEP 5: Build ProBERT tensors ===")
        if self._tcr is None:
            print("[SKIP] ProBERT tensor build – pipeline_TCR not importable.")
            return None

        concat = self._tcr.build_tensors(
            tcr_csv      = tcr_csv,
            trd_emb_path = trd_pkl,
            trg_emb_path = trg_pkl,
            emb_type     = "probert",
            out_prefix   = out_prefix,
            leiden_cols  = leiden_cols,                    # ← forwarded
        )
        print(f"  ProBERT tensors → {self.output_root}")
        return concat

    def _probert_embed(
        self,
        data_path: str,
        seq_col: str,
        model_path: str,
        output_pkl: str,
        chain_label: str = "",
    ) -> None:
        """
        Wrap pipeline_probert.run_eval() for one chain.
        """
        eval_tmp = str(self.output_root / f"probert_{chain_label.lower()}_eval")

        eval_args = argparse.Namespace(
            data_path        = data_path,
            seq_col          = seq_col,
            batch_size       = self.probert_batch_size,
            max_length       = self.probert_max_length,
            tokenizer        = "unirep",
            proteinbert_path = self.probert_base_path,
            model_path       = model_path,
            output_path      = eval_tmp,
        )

        if _probert_run_eval is None:
            raise ImportError(
                "pipeline_probert.run_eval is not available in this environment. "
                "Run in the ProBERT/TAPE environment or fix the import error shown at module import."
            )

        _probert_run_eval(eval_args)

        src = Path(eval_tmp) / "embeddings_eval.pkl"
        src.rename(output_pkl)
        print(f"  {chain_label} ProBERT embeddings → {output_pkl}")

    # =========================================================================
    # Concat inspector
    # =========================================================================
    @staticmethod
    def print_concat_summary(concat: Optional[List[dict]], max_rows: int = 3) -> None:
        """
        Print a human-readable summary of what is stored in a concat result
        returned by ``run()`` (i.e. ``results["esmc_concat"]`` or
        ``results["probert_concat"]``).

        Each record in the list is a dict produced by ``TCRPipeline.build_tensors``.
        This method shows:
          • total record count
          • all keys present in a record, labelled as [tensor] or [meta]
          • tensor shape, dtype, and basic statistics (min/max/mean)
          • a sample of the first ``max_rows`` records (non-tensor fields only)

        Parameters
        ----------
        concat : list of dict or None
            The concat result to inspect.
        max_rows : int
            How many sample records to print (default 3).
        """
        if concat is None:
            print("[print_concat_summary] concat is None – model may have been skipped.")
            return

        if not concat:
            print("[print_concat_summary] concat is an empty list.")
            return

        import torch

        print(f"\n{'='*60}")
        print(f"  Concat result summary  ({len(concat):,} records)")
        print(f"{'='*60}")

        # ── Key inventory from the first record ──────────────────────────────
        sample = concat[0]
        tensor_keys     = [k for k, v in sample.items() if isinstance(v, torch.Tensor)]
        non_tensor_keys = [k for k, v in sample.items() if not isinstance(v, torch.Tensor)]

        print(f"\n  All keys ({len(sample)}):")
        for k in sample:
            v = sample[k]
            if isinstance(v, torch.Tensor):
                print(f"    [tensor] {k:30s}  shape={tuple(v.shape)}  dtype={v.dtype}")
            else:
                print(f"    [meta  ] {k:30s}  type={type(v).__name__}  example={repr(v)[:60]}")

        # ── Tensor details ────────────────────────────────────────────────────
        if tensor_keys:
            print(f"\n  Tensor fields:")
            for k in tensor_keys:
                t = sample[k]
                print(f"    {k}: shape={tuple(t.shape)}, dtype={t.dtype}, "
                      f"min={t.min():.4f}, max={t.max():.4f}, mean={t.mean():.4f}")

        # ── Non-tensor (metadata) fields ──────────────────────────────────────
        print(f"\n  Metadata fields: {non_tensor_keys}")

        # ── Sample rows ───────────────────────────────────────────────────────
        n = min(max_rows, len(concat))
        print(f"\n  First {n} records (metadata only):")
        for i, rec in enumerate(concat[:n]):
            row = {k: rec[k] for k in non_tensor_keys}
            print(f"    [{i}] {row}")

        print(f"\n{'='*60}\n")


# =============================================================================
# CLI  –  python -m gdtcr.pipeline_embedding --help
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build ESMC and/or ProBERT TCR embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
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

    p.add_argument("--run_esmc",    dest="run_esmc",    action="store_true",  default=True)
    p.add_argument("--no_esmc",     dest="run_esmc",    action="store_false")
    p.add_argument("--run_probert", dest="run_probert", action="store_true",  default=True)
    p.add_argument("--no_probert",  dest="run_probert", action="store_false")

    p.add_argument("--esmc_chunk_size",    type=int, default=256)
    p.add_argument("--probert_batch_size", type=int, default=64)
    p.add_argument("--probert_max_length", type=int, default=45)
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    pipe = EmbeddingPipeline(
        output_root        = args.output_root,
        esmc_trd_weights   = args.esmc_trd_weights,
        esmc_trg_weights   = args.esmc_trg_weights,
        probert_trd_model  = args.probert_trd_model,
        probert_trg_model  = args.probert_trg_model,
        probert_base_path  = args.probert_base_path,
        leiden_cols        = args.leiden_cols,             # ← NEW
        esmc_chunk_size    = args.esmc_chunk_size,
        probert_batch_size = args.probert_batch_size,
        probert_max_length = args.probert_max_length,
    )

    results = pipe.run(
        h5ad_path     = args.h5ad_path,
        tcr_csv       = args.tcr_csv,
        cluster_col   = args.cluster_col,                  # ← NEW (None → auto)
        batch_col     = args.batch_col,
        target_batch  = args.target_batch,
        keep_clusters = args.keep_clusters,
        merge_rules   = [],
        out_prefix    = args.out_prefix,
        run_esmc      = args.run_esmc,
        run_probert   = args.run_probert,
        # leiden_cols not passed here → inherits from pipe.leiden_cols set above
    )

    # Print concat summaries so the user can confirm what landed in each .pt
    EmbeddingPipeline.print_concat_summary(results["esmc_concat"])
    EmbeddingPipeline.print_concat_summary(results["probert_concat"])


if __name__ == "__main__":
    main()