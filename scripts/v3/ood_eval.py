#!/usr/bin/env python3
"""ood_eval.py

Score the 30 held-out orphans with a trained OpenBinder model.  All 30 are expected
to be classified as non-binders (label=0) by the OpenBinder pipeline — they were set
aside from training precisely as an out-of-distribution stress test.

The held-out list lives in ``data/held_out_orphans.tsv``.  Each orphan's row
must exist in the feature CSVs — if any orphan is missing we flag it and skip
scoring that row.

Reports
-------
* Per-file prediction score (saved to the output CSV).
* Summary printed to stdout:
    - mean / median / max prediction score
    - false-positive rate (fraction scored >= 0.5)

Usage
-----
    python scripts/v3/ood_eval.py \\
        --config configs/rf_rest.yaml \\
        --model models/runs/rf_rest_<ts>/model.joblib \\
        --output models/runs/rf_rest_<ts>/ood_orphan_scores.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datamodule import DataModule, resolve_path  # noqa: E402
from feature_combine import assemble_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True,
                   help="Path to model.joblib (RF) or model.pt (MLP).")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Binder threshold for FPR computation (default 0.5).")
    return p.parse_args()


def score_rf(model_path: Path, X: np.ndarray) -> np.ndarray:
    pipe = joblib.load(model_path)
    return pipe.predict_proba(X)[:, 1]


def score_mlp(model_path: Path, X: np.ndarray, config: dict) -> np.ndarray:
    """Load an MLP checkpoint and score.  Relies on the checkpoint carrying
    scalers + hidden_dims so we can reconstruct the architecture."""
    import torch
    import torch.nn as nn

    ckpt = torch.load(model_path, map_location="cpu")
    mode = ckpt["mode"]
    hidden = ckpt["hidden_dims"]
    dropout = ckpt["dropout"]

    if mode == "both":
        # Two-branch; caller should have passed the concatenated matrix only if
        # assemble_matrix flat path was used.  For the production scaffold we
        # keep OOD MLP scoring simple and only support the single-branch flat
        # case.  The LOO harness's "both" path uses the same flat matrix.
        # Future work: load rest/unrest/extras separately.  For now, warn.
        raise NotImplementedError("ood_eval MLP two-branch mode not yet implemented")

    n_in = X.shape[1]
    layers: list[nn.Module] = []
    prev = n_in
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Apply stored scaler
    scalers = ckpt.get("scalers", {})
    scale_info = scalers.get("x")
    if scale_info is not None:
        mean = np.asarray(scale_info["mean"], dtype=np.float32)
        scale = np.asarray(scale_info["scale"], dtype=np.float32)
        X = (X - mean) / scale

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)).squeeze(-1)
        probs = torch.sigmoid(logits).numpy()
    return probs


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dm = DataModule.from_config(args.config)
    cfg = dm.config
    mode = cfg["feature_mode"]
    esm_dims = int(cfg.get("esm_pca_dims", 64))

    # Load features for the orphan set.  Easiest path: load the full merged
    # cohort (including held-out orphans by temporarily NOT applying the
    # hold-out filter), then subset to only the held-out files.
    raw = dm.load_features()
    if cfg.get("hold_out", {}).get("drop_error_rows", True):
        raw = dm.drop_error_rows(raw)
    raw = dm.attach_esm_pca(raw, fit=True)

    held_path = resolve_path(cfg["hold_out"]["held_out_orphans_tsv"], dm.project_root)
    held_df = pd.read_csv(held_path, sep="\t")
    held_files = list(held_df["file"].astype(str))

    subset = raw[raw["file"].isin(held_files)].copy()
    missing = sorted(set(held_files) - set(subset["file"].astype(str)))
    if missing:
        print(f"[ood_eval] WARNING: {len(missing)} held-out orphan(s) missing from cohort "
              f"(typically because a feature CSV does not cover them yet): {missing}", flush=True)

    if len(subset) == 0:
        print("[ood_eval] ERROR: no held-out orphans matched the cohort — nothing to score.",
              file=sys.stderr)
        sys.exit(1)

    X, feature_cols = assemble_matrix(subset, mode, esm_dims)
    print(f"[ood_eval] scoring {len(subset)} orphans in {X.shape[1]}-dim feature space", flush=True)

    if cfg["model_type"] == "rf":
        probs = score_rf(args.model, X)
    else:
        probs = score_mlp(args.model, X, cfg)

    out = pd.DataFrame({
        "file": subset["file"].astype(str).values,
        "pred_prob": probs,
        "classified_binder": (probs >= args.threshold).astype(int),
    })
    out.to_csv(args.output, index=False)

    fpr = float((probs >= args.threshold).mean())
    summary = {
        "n_scored": int(len(probs)),
        "n_missing_from_cohort": int(len(missing)),
        "threshold": float(args.threshold),
        "mean_score": float(np.mean(probs)),
        "median_score": float(np.median(probs)),
        "max_score": float(np.max(probs)),
        "false_positive_rate": fpr,
        "missing_files": missing,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[ood_eval] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[ood_eval] scores → {args.output}", flush=True)
    print(f"[ood_eval] summary → {summary_path}", flush=True)


if __name__ == "__main__":
    main()
