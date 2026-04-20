#!/usr/bin/env python3
"""rf_train.py

Train an OpenBinder Random Forest on the full (non-LOO) training set for one config.

Writes to ``--output-dir``:
    model.joblib             — fitted (scaler, RF) pipeline
    metrics.json             — AUROC / AUPRC / accuracy on stratified hold-out
    predictions.csv          — file, label, pred_prob (per-sample)
    feature_importances.csv  — sorted Gini importances
    meta.json                — config snapshot + cohort sizes

Usage
-----
    python scripts/v3/rf_train.py \\
        --config configs/rf_rest.yaml \\
        --output-dir models/runs/rf_rest_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # so sibling modules import cleanly

from datamodule import DataModule  # noqa: E402
from feature_combine import assemble_matrix  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="Fraction of cohort used for a stratified validation split (default 0.15).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_rf(cfg: dict) -> RandomForestClassifier:
    rf_cfg = cfg.get("rf", {})
    return RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 500)),
        max_depth=rf_cfg.get("max_depth", None),
        min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=int(rf_cfg.get("random_state", 42)),
        n_jobs=-1,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rf_train] config={args.config}  output-dir={args.output_dir}", flush=True)
    dm = DataModule.from_config(args.config)
    cfg = dm.config

    if cfg["model_type"] != "rf":
        raise SystemExit(f"rf_train expects model_type=rf, got {cfg['model_type']!r}")

    df = dm.prepare()
    esm_dims = int(cfg.get("esm_pca_dims", 64))
    X, feature_cols = assemble_matrix(df, cfg["feature_mode"], esm_dims)
    y = dm.build_labels(df)
    files = df["file"].astype(str).values

    print(f"[rf_train] cohort: N={len(y)} pos={int(y.sum())} neg={int((y==0).sum())} feats={X.shape[1]}", flush=True)

    # Stratified hold-out split for metric reporting (the LOO harness handles the per-fold story)
    X_tr, X_va, y_tr, y_va, files_tr, files_va = train_test_split(
        X, y, files,
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=y,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", build_rf(cfg)),
    ])
    pipe.fit(X_tr, y_tr)

    probs_va = pipe.predict_proba(X_va)[:, 1]
    preds_va = (probs_va >= 0.5).astype(int)

    metrics = {
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_va)),
        "n_features": int(X.shape[1]),
        "auroc_val": float(roc_auc_score(y_va, probs_va)),
        "auprc_val": float(average_precision_score(y_va, probs_va)),
        "accuracy_val": float(accuracy_score(y_va, preds_va)),
    }

    # Refit on full cohort for production artifact
    pipe_full = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", build_rf(cfg)),
    ])
    pipe_full.fit(X, y)

    # ── Write artifacts ──────────────────────────────────────────────────
    joblib.dump(pipe_full, args.output_dir / "model.joblib")

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    probs_all = pipe_full.predict_proba(X)[:, 1]
    pd.DataFrame({
        "file": files,
        "label": y,
        "pred_prob": probs_all,
    }).to_csv(args.output_dir / "predictions.csv", index=False)

    importances = pipe_full.named_steps["rf"].feature_importances_
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp_df.to_csv(args.output_dir / "feature_importances.csv", index=False)

    meta = {
        "config_name": cfg["config_name"],
        "model_type": cfg["model_type"],
        "feature_mode": cfg["feature_mode"],
        "esm_pca_dims": esm_dims,
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "feature_cols": feature_cols,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[rf_train] metrics: {json.dumps(metrics, indent=2)}", flush=True)
    print(f"[rf_train] artifacts written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
