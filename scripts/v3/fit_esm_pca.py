#!/usr/bin/env python3
"""fit_esm_pca.py — Fit and save the ESM PCA transform from the training cohort.

This script is a lightweight wrapper that runs the DataModule pipeline for the
both_all config (which has ESM enabled) and saves the fitted PCA bundle to
models/checkpoints/esm_pca.joblib.

The saved bundle is required by score.py at inference time.  It replaces the
previous behaviour where score.py re-fitted the PCA from training CSVs,
which is incorrect because it produced a different embedding space than the
one used during model training (Bug 7 fix).

Usage
-----
    python scripts/v3/fit_esm_pca.py
    python scripts/v3/fit_esm_pca.py --config configs/mlp_both_all.yaml
    python scripts/v3/fit_esm_pca.py --output models/checkpoints/esm_pca.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from datamodule import DataModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "mlp_both_all.yaml",
        help="YAML config to use (must have feature_mode that includes ESM). "
             "Default: configs/mlp_both_all.yaml",
    )
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "models" / "checkpoints" / "esm_pca.joblib",
        help="Destination path for the PCA bundle. "
             "Default: models/checkpoints/esm_pca.joblib",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[fit_esm_pca] config={args.config}", flush=True)
    print(f"[fit_esm_pca] output={args.output}", flush=True)

    dm = DataModule.from_config(args.config)
    # prepare() now automatically saves esm_pca.joblib when save_esm_pca is given
    dm.prepare(save_esm_pca=args.output)
    print(f"[fit_esm_pca] Done. ESM PCA saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
