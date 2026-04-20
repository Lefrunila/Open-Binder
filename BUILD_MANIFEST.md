# BUILD_MANIFEST

File-by-file inventory of what ships in this repository, grouped by
top-level directory, with a one-line rationale per file. Use this as the
authoritative list of what's in the OpenBinder reproducibility bundle.

Generated: 2026-04-19.

## Root

| Path | Purpose |
|---|---|
| `README.md` | Project overview, results table, installation, reproduce-from-scratch steps, feature schema. |
| `LICENSE` | MIT licence. |
| `CITATION.cff` | Citation metadata (paper venue TBD). |
| `BUILD_MANIFEST.md` | This file. |
| `environment.yml` | Full conda environment (recommended — includes OpenMM / PDBFixer / mdtraj for feature re-extraction). |
| `requirements.txt` | Pip requirements (subset of the conda env; sufficient for OpenBinder training and inference). |
| `.gitignore` | Ignores `*.pdb`, `__pycache__`, `*.bak*`, training-run outputs, etc. Exceptions for `examples/9YC3_relaxed.pdb` and `data/structures/*.tar.gz`. |
| `.gitattributes` | Git LFS tracking directive for `data/structures/*.tar.gz` (combined payload ≈1.1 GB). |

## configs/

One YAML per final model. Each config declares feature sources, feature
mode, hyperparameters, and random seed.

| Path | Purpose |
|---|---|
| `configs/rf_rest.yaml` | Random Forest on the 95-feature rest-only set (27 openmm_rest + 4 cocada_rest + 64 esm). |
| `configs/rf_unrest.yaml` | Random Forest on the 95-feature unrest-only set. |
| `configs/rf_both_delta.yaml` | Random Forest on 122 features (rest + openmm delta + cocada_rest + esm). |
| `configs/rf_both_raw.yaml` | Random Forest on 126 features (rest + unrest + both cocada + esm). |
| `configs/rf_both_all.yaml` | Random Forest on the full 153-feature combined set (champion RF). |
| `configs/mlp_both_all.yaml` | SingleBranchMLP on the full 153-feature combined set (champion MLP). |

## data/

Feature CSVs and sidecar TSVs. Every CSV is keyed by the antigen stem (or
negative sample ID) and aligns 1:1 with the structure tarballs shipped
under `data/structures/`.

| Path | Rows (incl. header) | Purpose |
|---|---:|---|
| `data/features_positives_openmm_v2.csv` | 1,130 | 27 OpenMM interface features per positive complex, CA-restrained relax. |
| `data/features_positives_openmm_unrest.csv` | 1,130 | 27 OpenMM interface features per positive complex, unrestrained relax. |
| `data/features_negatives_openmm_v2.csv` | 2,289 | 27 OpenMM interface features per negative complex, CA-restrained relax. |
| `data/features_negatives_openmm_unrest.csv` | 2,289 | 27 OpenMM interface features per negative complex, unrestrained relax. |
| `data/cocada_positives_v2.csv` | 1,130 | 4 COCaDA contact-level features per positive, CA-restrained relax. |
| `data/cocada_positives_unrest.csv` | 1,130 | 4 COCaDA contact-level features per positive, unrestrained relax. |
| `data/cocada_negatives_v2.csv` | 2,289 | 4 COCaDA contact-level features per negative, CA-restrained relax. |
| `data/cocada_negatives_unrest.csv` | 2,289 | 4 COCaDA contact-level features per negative, unrestrained relax. |
| `data/esm_ppi_positives.csv` | 1,130 | ESM-PPI sequence embedding per positive VHH (later PCA-reduced to 64 components in `datamodule.py`). |
| `data/esm_ppi_negatives.csv` | 2,289 | ESM-PPI sequence embedding per negative VHH. |
| `data/held_out_orphans.tsv` | 31 | 30 held-out negative orphans used exclusively for the OOD stress test. |
| `data/orphan_training_assignments.tsv` | 6 | Provenance sidecar recording how the orphan hold-out was partitioned. |

## data/structures/

Six gzipped tarballs shipping the full 10,162-PDB set for every relaxation
state. Tracked via Git LFS.

| Path | PDB count | Compressed | Uncompressed (approx) | Contents |
|---|---:|---:|---:|---|
| `data/structures/positives_cleaned.tar.gz` | 1,129 |  81 MB | 1.5 GB | Pre-relax cleaned positives (input to OpenMM). Source: `NanoBinder-RF-v2/data/positives/`. |
| `data/structures/positives_relaxed.tar.gz` | 1,129 | 136 MB | 2.7 GB | CA-restrained OpenMM relax of positives. Source: `NanoBinder-RF-v2/data/positives_relaxed/`. |
| `data/structures/positives_unrest.tar.gz` | 1,129 | 136 MB | 652 MB | Unrestrained OpenMM relax of positives. Source: `NanoBinder-RF-v2/data/positives_unrest/`. |
| `data/structures/negatives_cleaned.tar.gz` | 2,288 | 162 MB | 2.2 GB | Pre-relax cleaned negatives. Source: `NanoBinder-RF-v2/data/negatives/`. |
| `data/structures/negatives_relaxed.tar.gz` | 2,288 | 269 MB | 6.7 GB | CA-restrained OpenMM relax of negatives. Source: `NanoBinder-RF-v2/data/negatives_relaxed/`. |
| `data/structures/negatives_unrest.tar.gz` | 2,288 | 271 MB | 1.3 GB | Unrestrained OpenMM relax of negatives. Source: `NanoBinder-RF-v2/data/negatives_unrest/`. |
| `data/structures/README.md` | — | — | — | Extraction instructions, layout diagram, naming conventions, orphan hold-out note. |

Totals: 10,162 PDB files; ≈1.1 GB compressed; ≈15 GB uncompressed.

## scripts/v3/

Training, evaluation, and LOO harness. Pure Python; no CLI runtime
dependencies on OpenMM / PDBFixer.

| Path | Lines | Purpose |
|---|---:|---|
| `scripts/v3/__init__.py` | 21 | Package marker + public re-exports. |
| `scripts/v3/datamodule.py` | 338 | Load feature CSVs, build the 3,387-sample cohort, apply the 30-orphan hold-out, run PCA on ESM. |
| `scripts/v3/feature_combine.py` | 175 | Column-family helpers for rest / unrest / both_delta / both_raw / both_all modes. |
| `scripts/v3/rf_train.py` | 160 | Train a single Random Forest on the full cohort from a YAML config. |
| `scripts/v3/mlp_train.py` | 393 | Train the SingleBranchMLP (PyTorch) on the full cohort, early stopping, writes checkpoint + training curve. |
| `scripts/v3/loo_harness.py` | 429 | Leave-one-out benchmark over the 1,129 antigen stems — per-fold JSON + pooled metrics. |
| `scripts/v3/loo_compare.py` | 133 | Aggregate multiple LOO runs into a comparison table (produces the headline results table). |
| `scripts/v3/ood_eval.py` | 171 | Score the 30 held-out orphans using a saved checkpoint. |

## models/checkpoints/

One trained artifact per config. Each checkpoint directory carries the
model, its training metadata, per-sample predictions on the full cohort,
and feature importances (RFs only).

| Path | Size | Purpose |
|---|---:|---|
| `models/checkpoints/rf_rest/model.joblib` | ~29 MB | Trained `rf_rest` sklearn model. |
| `models/checkpoints/rf_rest/meta.json` | — | Feature list, seed, hyperparameters, training timestamp. |
| `models/checkpoints/rf_rest/metrics.json` | — | Full-cohort training metrics. |
| `models/checkpoints/rf_rest/predictions.csv` | — | Per-sample scores on the 3,387-row training cohort. |
| `models/checkpoints/rf_rest/feature_importances.csv` | — | Sklearn impurity-based importances. |
| `models/checkpoints/rf_unrest/*` | ~29 MB total | Same layout as `rf_rest`, but for unrest features. |
| `models/checkpoints/rf_both_delta/*` | ~24 MB total | Same layout, 122-feature both_delta model. |
| `models/checkpoints/rf_both_raw/*` | ~27 MB total | Same layout, 126-feature both_raw model. |
| `models/checkpoints/rf_both_all/*` | ~24 MB total | Champion RF — full 153-feature combined set. |
| `models/checkpoints/mlp_both_all/model.pt` | — | PyTorch state dict for the SingleBranchMLP champion. |
| `models/checkpoints/mlp_both_all/meta.json` | — | Architecture, optimizer, batch size, seed, best-epoch. |
| `models/checkpoints/mlp_both_all/metrics.json` | — | Final metrics on the cohort. |
| `models/checkpoints/mlp_both_all/predictions.csv` | — | Per-sample scores. |
| `models/checkpoints/mlp_both_all/training_curve.csv` | — | Loss / AUROC per epoch (for the training-curve figure). |

## models/loo/

Per-model LOO output. Each model directory contains a `per_fold/`
subdirectory with 1,129 JSON files (one per antigen stem), a pooled
metrics summary, and a per-antigen pass-rate report.

| Path | Per-fold files | Purpose |
|---|---:|---|
| `models/loo/rf_rest/` | 1,129 | LOO output for `rf_rest` — reproduces row 1 of the results table. |
| `models/loo/rf_unrest/` | 1,129 | LOO output for `rf_unrest` — reproduces row 2. |
| `models/loo/rf_both_delta/` | 1,129 | LOO output for `rf_both_delta` — reproduces row 3. |
| `models/loo/rf_both_raw/` | 1,129 | LOO output for `rf_both_raw` — reproduces row 4. |
| `models/loo/rf_both_all/` | 1,129 | LOO output for the champion RF — reproduces row 5. |
| `models/loo/mlp_both_all/` | 1,129 | LOO output for the champion MLP — reproduces row 6. |
| `models/loo/*/pooled_metrics.json` | — | AUROC / TPR / TNR / AUPRC pooled across all held-out predictions. |
| `models/loo/*/pass_rate_by_antigen.csv` | — | Per-antigen pass rate (for diagnosis of hard antigens). |
| `models/loo/*/per_fold/<STEM>.json` | — | One JSON per antigen fold: held-out IDs, scores, labels, fold-level metrics. |

## examples/

| Path | Size | Purpose |
|---|---:|---|
| `examples/9YC3_relaxed.pdb` | 300 KB | Single relaxed positive shipped as a schema / tooling example. Excepted from the `.gitignore` `*.pdb` rule. |

## docs/

Empty — reserved for supplementary writeups. Present so downstream docs
tooling does not have to `mkdir` it.
