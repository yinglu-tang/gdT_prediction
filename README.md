# gdTCR

gdTCR: Gamma-delta TCR sequence embedding and cluster prediction

## Introduction

Gamma-delta T cell receptors contain a delta chain (TRD) and a gamma chain (TRG). Their CDR3 amino acid sequences and V/J gene annotations provide information for characterizing receptor repertoires and comparing new samples with a reference dataset.

The gdTCR package uses fine-tuned ESMC and ProBERT models to extract CDR3 sequence embeddings from both chains. These embeddings, together with V/J annotations, are used by trained classifiers to predict cluster labels for new records. The workflow supports paired TRD/TRG records and bulk repertoire records with only one observed chain.

## Graphical abstract

![image](img/workflow.png)

## Installation

We recommend using conda to manage the Python environment:

```bash
conda create --name gdtcr python=3.10
conda activate gdtcr
```

Then, install the package from the project folder containing `pyproject.toml`:

```bash
cd /path/to/dir
pip install ".[all]"
```

This command installs gdTCR and its dependencies for ESMC, ProBERT, and the example notebook together.

#### Note:

Input datasets and trained model weights are not included in the package. Before running the examples, prepare the TRD/TRG ESMC weights, TRD/TRG ProBERT weights, the pretrained ProteinBERT directory, and the classifier run directory described below. ESMC also loads its pretrained ESMC-600M backbone, which may require a download on first use.

## Test example

To check embedding extraction and prediction, run the quick-start example with your input data and model files.

We have included an [example notebook](notebook/example.ipynb) covering embedding extraction and prediction for reference, bulk, MiXCR, and TRUST4 inputs. Open it with:

```bash
jupyter lab notebook/example.ipynb
```

Use a kernel from the environment where gdTCR is installed. The notebook's paths are relative to `notebook/`; replace the data and model paths before running the cells.

## Usage

### Quick start

Once the input data have been processed into the supported format, the full workflow can be run from Python. The following example extracts embeddings from both models and combines their classifier outputs to predict cluster labels.

Run this example from the project folder and replace the input and model paths with your own files:

```python
from gdtcr import EmbeddingPipeline

pipe = EmbeddingPipeline(
    output_root="output/example",
    esmc_trd_weights="models/esmc_trd/model.pt",
    esmc_trg_weights="models/esmc_trg/model.pt",
    probert_trd_model="models/probert_trd/model.pth",
    probert_trg_model="models/probert_trg/model.pth",
    probert_base_path="models/proteinbert",
)

results = pipe.run(
    tcr_csv="data/tcr_cleaned.csv",
    out_prefix="example",
    cell_id_col="cell_id",
    sample_id_col="sample_id",
    run_esmc=True,
    run_probert=True,
)

pred_csv = pipe.predict_from_concat(
    model_path="models/prediction_run",
    emb_type="fused",
    esmc_concat=results["esmc_concat"],
    probert_concat=results["probert_concat"],
    out_prefix="example",
    alpha=0.5,
)

print(pred_csv)
```

The prediction table is saved to:

```text
output/example/example_fused_predictions.csv
```

### Data preparation

The CSV workflow takes a TCR table with one row per cell or bulk repertoire record. An example of the input format is shown below. These sequences are illustrative.

| cell_id | sample_id | trd_sequence | TRDV | TRDJ | trg_sequence | TRGV | TRGJ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cell_001 | sample_A | CALGELGDDKLIF | TRDV1 | TRDJ1 | CATWDTTGWFKIF | TRGV9 | TRGJP |
| cell_002 | sample_A | CACDTGGYTDKLIF | TRDV2 | TRDJ1 | | | |
| cell_003 | sample_B | | | | CATWDRPEKLF | TRGV9 | TRGJ1 |

Save the table as `data/tcr_cleaned.csv`, with commas as the delimiter.

Sequence columns:

`trd_sequence` and `trg_sequence` contain the CDR3 amino acid sequences of the delta and gamma chains. Supply at least one observed chain per record for sequence-based prediction. Leave the other chain blank when it is unavailable.

Identifier columns:

`cell_id` identifies each row and should be unique, including for bulk records. `sample_id` identifies the sample and can be shared by multiple rows. Use `cell_id_col` and `sample_id_col` to select differently named columns in your input.

V/J columns:

`TRDV`, `TRDJ`, `TRGV`, and `TRGJ` contain the V/J annotations used by the joint classifiers. Their gene or allele naming should match the classifier vocabulary.

Optional columns:

`dataset` identifies the source dataset. Existing cluster annotations can be retained by specifying `leiden_cols`. New input records do not need known cluster labels for CSV-based prediction.

MiXCR and TRUST4 output must be converted to this table format before running gdTCR. The example notebook uses already-prepared CSVs. For bulk data, TRD-only and TRG-only observations can remain separate rows, with the unobserved chain left blank.

### Model preparation

The quick-start example requires fine-tuned weights for both chains and both embedding models. The paths are supplied through `esmc_trd_weights`, `esmc_trg_weights`, `probert_trd_model`, and `probert_trg_model`. `probert_base_path` points to the pretrained ProteinBERT directory used by TAPE.

For fused prediction, the classifier run directory should contain:

```text
models/prediction_run/
    hparams.json
    vj_vocab.json
    label_map.json
    best_model_esmc.pt
    best_model_probert.pt
```

`hparams.json` stores the classifier configuration. `vj_vocab.json` stores the V/J vocabulary. `label_map.json` maps model class indices to the original cluster labels; if it is absent, the output uses class indices. The vocabulary and label mapping may also be stored in the parent of the classifier run directory.

### Feature extraction

Feature extraction is performed automatically by `pipe.run()`. gdTCR first extracts the unique CDR3 sequences for each chain, computes their ESMC and ProBERT embeddings, and matches the embeddings back to the input rows. The delta and gamma chain embeddings are concatenated in that order to form one representation per record.

When a chain is missing or its embedding is unavailable, the corresponding vector is filled with zeros. The output metadata marks each chain as `observed` or `missing`, and a missing-chain summary is saved for each sample.

To inspect the generated records:

```python
pipe.print_concat_summary(results["esmc_concat"])
pipe.print_concat_summary(results["probert_concat"])
```

### Prediction models: `https://zenodo.org/records/22776222?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijc0YzMxMzY3LTQ2N2YtNGZlYy04NzQ2LWMzZTk5YzA1OWEzNCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MzEyYmIxZGJlMzE5YzRhN2ZhYWUxNzMwNWExMjQ2NiJ9.VqRfGi7nRkA9yqJi7P9b6tQWEotOkkWlCiIKBj9AL9uqbukOeRuP53VG8w8efF9UQGxUP6Zqn0q1PPdJAHQdug`

Four prediction modes are supported:

| `emb_type` | Input features | Required checkpoints |
| --- | --- | --- |
| `esmc` | ESMC embeddings and V/J | `best_model_esmc.pt` |
| `probert` | ProBERT embeddings and V/J | `best_model_probert.pt` |
| `vj` | V/J annotations | `best_model_vj.pt` |
| `fused` | Combined ESMC and ProBERT predictions, each using V/J | `best_model_esmc.pt`, `best_model_probert.pt` |

For V/J prediction, provide either concat input as the source of metadata:

```python
pred_csv = pipe.predict_from_concat(
    model_path="models/prediction_run",
    emb_type="vj",
    esmc_concat=results["esmc_concat"],
    out_prefix="example",
)
```

Fused prediction requires both concat inputs, matched by `cell_id`. Only overlapping records are used. Each mode loads its named checkpoint.

### Reusing existing embeddings

If the embeddings have already been generated, the saved concat files can be passed directly to prediction:

```python
pred_csv = pipe.predict_from_concat(
    model_path="models/prediction_run",
    emb_type="fused",
    esmc_concat="output/example/example_esmc_concat.pt",
    probert_concat="output/example/example_probert_concat.pt",
    out_prefix="example_rerun",
    alpha=0.5,
)
```

### Output files

All results are saved under `output_root`. The quick-start example produces the following main files:

| File | Description |
| --- | --- |
| `example_computed_metadata.csv` | Input metadata with prepared identifiers and TCR annotations. |
| `example_trd_unique_sequences.csv`, `example_trg_unique_sequences.csv` | Unique CDR3 sequences for each chain. |
| `example_trd_esmc_embeddings.pt`, `example_trg_esmc_embeddings.pt` | ESMC sequence embeddings. |
| `example_trd_probert.pkl`, `example_trg_probert.pkl` | ProBERT sequence embeddings. |
| `example_esmc_concat.pt`, `example_probert_concat.pt` | Per-record metadata and concatenated TRD/TRG embeddings. |
| `example_esmc_X.npy`, `example_probert_X.npy` | Numeric embedding matrices. |
| `example_esmc_metadata.csv`, `example_probert_metadata.csv` | Metadata corresponding to the embedding matrix rows. |
| `example_esmc_missing_chain_handling.csv`, `example_probert_missing_chain_handling.csv` | Chain availability for each record. |
| `example_esmc_missing_chain_summary_by_sample.csv`, `example_probert_missing_chain_summary_by_sample.csv` | Missing-chain counts and fractions for each sample. |
| `example_fused_predictions.csv` | Predicted cluster labels and class probabilities. |

An example prediction table is shown below, displaying selected columns for a hypothetical three-class model. The numbers illustrate the output format and are not results from the sequences above.

| cell_id | sample_id | pred_cluster_idx | pred_cluster_label | pred_confidence | prob_cluster_1 | prob_cluster_2 | prob_cluster_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cell_001 | sample_A | 0 | 1 | 0.86 | 0.86 | 0.09 | 0.05 |
| cell_002 | sample_A | 1 | 2 | 0.72 | 0.18 | 0.72 | 0.10 |
| cell_003 | sample_B | 2 | 3 | 0.81 | 0.07 | 0.12 | 0.81 |

`pred_cluster_idx` is the model class index. `pred_cluster_label` is the corresponding cluster label after applying the label mapping. `pred_confidence` is the highest softmax probability, and `prob_cluster_<label>` reports the probability for each class. The actual labels and number of probability columns depend on your trained classifier.

Read the prediction table in Python with:

```python
import pandas as pd

predictions = pd.read_csv("output/example/example_fused_predictions.csv")
print(predictions[["cell_id", "sample_id", "pred_cluster_label", "pred_confidence"]].head())
```

The missing-chain summary for the illustrative input would look as follows if all supplied sequences were embedded successfully. Here, `dataset` defaults to `sample_id`:

| dataset | sample_id | n_cells | n_trd_missing | n_trg_missing | frac_trd_missing | frac_trg_missing |
| --- | --- | --- | --- | --- | --- | --- |
| sample_A | sample_A | 2 | 0 | 1 | 0.0 | 0.5 |
| sample_B | sample_B | 1 | 1 | 0 | 1.0 | 0.0 |

For bulk input, `n_cells` counts input records. Additional input fields such as read counts remain in the computed metadata CSV; join them to the prediction table by unique `cell_id` when needed.

### List of Parameters

The main Python parameters are listed below:

| Parameter | Description | Default |
| --- | --- | --- |
| `output_root` | Directory for all output files. | Required |
| `esmc_trd_weights`, `esmc_trg_weights` | Fine-tuned ESMC weights for each chain. | Supply when using ESMC |
| `probert_trd_model`, `probert_trg_model` | Fine-tuned ProBERT weights for each chain. | Supply when using ProBERT |
| `probert_base_path` | Pretrained ProteinBERT directory. | `../models/proteinbert` |
| `tcr_csv` | Prepared input TCR table. | Supply an input path |
| `out_prefix` | Prefix for output filenames. | `cohort3` in `run()`; `newdata` in prediction |
| `cell_id_col` | Column containing unique record identifiers. | Auto-detected |
| `sample_id_col` | Column containing sample identifiers. | Auto-detected |
| `dataset_col` | Column containing dataset identifiers. | Existing `dataset`, otherwise sample identifier |
| `reference` | Append `*01` to V/J names without an allele suffix during CSV preparation. | `False` |
| `run_esmc`, `run_probert` | Enable embedding extraction for each model. | Both `True` |
| `esmc_chunk_size` | Number of sequences per ESMC processing chunk. | `1000` |
| `probert_batch_size` | ProBERT inference batch size. | `64` |
| `probert_max_length` | ProBERT token-length limit. | `45` |
| `leiden_cols` | Existing cluster columns to preserve. | `["leiden_0.4"]` |
| `model_path` | Classifier run directory. | Required for prediction |
| `emb_type` | Prediction mode from the table above. | `fused` |
| `alpha` | ESMC weight in fused prediction. | Classifier configuration, otherwise `0.5` |
| `batch_size` | Classifier inference batch size. | `512` |
| `device` | Device used for classifier inference. | CUDA if available, otherwise CPU |
| `vj_mode` | V/J encoding as `gene` or `allele`. | Classifier configuration, otherwise `allele` |

A few additional notes for the parameters:

(1) `reference`: Set this to `True` only when adding the default `*01` suffix matches the reference classifier's annotation convention. Existing allele suffixes are retained. This option does not infer alleles from sequence, and setting it to `False` does not disable V/J features.

(2) `alpha`: Fused prediction combines the model logits as `alpha * ESMC + (1 - alpha) * ProBERT` before applying softmax. Use a value between 0 and 1; `0.5` gives equal weight to both models.

(3) `esmc_chunk_size` and `probert_batch_size`: ESMC processes sequences sequentially within each chunk, while ProBERT batches sequences together. Reduce the ProBERT batch size if GPU memory is insufficient.

(4) `h5ad_path`: An annotated AnnData file can be supplied instead of `tcr_csv`. TCR fields are read from `adata.obs`, and an existing `cluster_col` is required. `batch_col`, `target_batch`, `keep_clusters`, and `merge_rules` control filtering and cluster merging. Supply exactly one input source. For new records without known cluster annotations, use the CSV examples above.
