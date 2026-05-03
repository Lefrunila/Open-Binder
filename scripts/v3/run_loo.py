#!/usr/bin/env python3
"""run_loo.py

Runner for all 6 OpenBinder LOO configs. RF configs run in parallel
(multiprocessing); MLP runs after on GPU. ESM PCA is refit per fold inside
``loo_harness.py`` (no global-cohort leakage).

Outputs:
  models/loo_results/<config>/per_fold/*.json
  models/loo_results/<config>/pooled_metrics.json
  models/loo_results/<config>/pass_rate_by_antigen.csv
  results/loo_<config>.csv               (per-fold flat CSV)
  results/loo_summary.csv                (aggregate comparison table)

(Renamed from ``run_loo_connolly_sc.py`` after the Connolly SES feature became
the canonical sc implementation; the old "connolly_sc" suffix served only to
distinguish that intermediate run from the prior MSMS-based one.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PYTHON = os.environ.get("PYTHON", sys.executable)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Open-Binder/
SCRIPT_DIR = Path(__file__).resolve().parent
HARNESS = SCRIPT_DIR / "loo_harness.py"
COMPARE = SCRIPT_DIR / "loo_compare.py"

CONFIGS_DIR = PROJECT_ROOT / "configs"
LOO_BASE = PROJECT_ROOT / "models" / "loo_results"
RESULTS_DIR = PROJECT_ROOT / "results"

RF_CONFIGS = [
    "rf_rest",
    "rf_unrest",
    "rf_both_delta",
    "rf_both_raw",
    "rf_both_all",
]
MLP_CONFIGS = [
    "mlp_both_all",
]

LOG_DIR = PROJECT_ROOT / "results" / "logs"


def run_one_config(cfg_name: str, device: str | None = None) -> tuple[str, int]:
    """Run the LOO harness for one config. Returns (cfg_name, returncode)."""
    out_dir = LOO_BASE / cfg_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"loo_{cfg_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON,
        str(HARNESS),
        "--config", str(CONFIGS_DIR / f"{cfg_name}.yaml"),
        "--output-dir", str(out_dir),
    ]
    if device:
        cmd += ["--device", device]

    print(f"[runner] launching {cfg_name} → {log_path}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as fh:
        ret = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    print(f"[runner] {cfg_name} finished rc={ret} elapsed={elapsed:.1f}s", flush=True)
    return cfg_name, ret


def _worker(args):
    cfg_name, device = args
    return run_one_config(cfg_name, device)


def per_fold_to_csv(run_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Convert per-fold JSON files to a flat CSV with one row per sample."""
    per_fold_dir = run_dir / "per_fold"
    rows = []
    for f in sorted(per_fold_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            continue
        antigen = d["antigen"]
        pos_file = d["positive"]
        pos_prob = d["pos_prob"]
        neg_probs = d["neg_probs"]
        trio_size = d["trio_size"]
        rank = d["rank"]
        passed = d["passed"]
        # positive row
        rows.append({
            "antigen": antigen,
            "file": pos_file,
            "label": 1,
            "pred_prob": pos_prob,
            "pred_class": int(pos_prob >= 0.5),
            "rank_in_trio": rank,
            "trio_size": trio_size,
            "passed": int(passed),
        })
        for j, np_ in enumerate(neg_probs):
            rows.append({
                "antigen": antigen,
                "file": d.get("negatives", [f"neg_{j}"])[j] if j < len(d.get("negatives", [])) else f"neg_{antigen}_{j}",
                "label": 0,
                "pred_prob": np_,
                "pred_class": int(np_ >= 0.5),
                "rank_in_trio": None,
                "trio_size": trio_size,
                "passed": None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[runner] saved per-fold CSV: {out_csv} ({len(df)} rows)", flush=True)
    return df


def pool_metrics(run_dir: Path) -> dict:
    """Read per-fold JSONs and compute aggregate metrics."""
    per_fold_dir = run_dir / "per_fold"
    y_true = []
    y_prob = []
    pos_probs = []
    neg_probs_list = []
    passes = 0
    n_folds = 0
    skipped = 0

    for f in sorted(per_fold_dir.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("skipped"):
            skipped += 1
            continue
        n_folds += 1
        pp = d["pos_prob"]
        pos_probs.append(pp)
        y_true.append(1)
        y_prob.append(pp)
        for np_ in d["neg_probs"]:
            neg_probs_list.append(np_)
            y_true.append(0)
            y_prob.append(np_)
        passes += int(d["passed"])

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    preds = (y_prob >= 0.5).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (tpr + tnr)

    pos_probs_arr = np.asarray(pos_probs)
    neg_probs_arr = np.asarray(neg_probs_list)
    n_pos_fail = int((pos_probs_arr < 0.5).sum())
    n_neg_fail = int((neg_probs_arr >= 0.5).sum())
    borderline_pos = int(((pos_probs_arr >= 0.4) & (pos_probs_arr < 0.5)).sum())
    confident_neg = int((neg_probs_arr > 0.7).sum())

    return {
        "config": run_dir.name,
        "n_folds": n_folds,
        "n_skipped": skipped,
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "pass_rate": passes / max(n_folds, 1),
        "pooled_auroc": float(roc_auc_score(y_true, y_prob)),
        "pooled_auprc": float(average_precision_score(y_true, y_prob)),
        "pooled_accuracy": float((preds == y_true).mean()),
        "tpr_at_0.5": float(tpr),
        "tnr_at_0.5": float(tnr),
        "bal_acc": float(bal_acc),
        "pos_mean_prob": float(np.mean(pos_probs)),
        "neg_mean_prob": float(np.mean(neg_probs_list)),
        "n_pos_fail": n_pos_fail,
        "n_neg_fail": n_neg_fail,
        "borderline_pos_fail": borderline_pos,
        "borderline_pos_fail_frac": float(borderline_pos / max(n_pos_fail, 1)),
        "confident_neg_fail": confident_neg,
        "confident_neg_fail_frac": float(confident_neg / max(n_neg_fail, 1)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: RF configs in parallel ---
    print(f"[runner] Phase 1: running {len(RF_CONFIGS)} RF configs in parallel (nproc=5)...", flush=True)
    rf_args = [(cfg, None) for cfg in RF_CONFIGS]
    t0 = time.time()
    with Pool(processes=5) as pool:
        rf_results = pool.map(_worker, rf_args)
    print(f"[runner] All RF configs finished in {time.time()-t0:.1f}s", flush=True)
    for cfg, rc in rf_results:
        if rc != 0:
            print(f"[runner] WARNING: {cfg} exited with rc={rc}", flush=True)

    # --- Phase 2: MLP config on GPU ---
    print(f"[runner] Phase 2: running MLP configs...", flush=True)
    mlp_results = []
    for cfg in MLP_CONFIGS:
        cfg_name, rc = run_one_config(cfg, device="cuda")
        mlp_results.append((cfg_name, rc))
        if rc != 0:
            print(f"[runner] WARNING: {cfg} exited with rc={rc}", flush=True)

    # --- Phase 3: compile per-fold CSVs and summary ---
    print(f"[runner] Phase 3: compiling metrics...", flush=True)
    all_configs = RF_CONFIGS + MLP_CONFIGS
    summary_rows = []
    for cfg in all_configs:
        run_dir = LOO_BASE / cfg
        if not (run_dir / "per_fold").exists():
            print(f"[runner] WARNING: {cfg} per_fold dir missing, skipping metrics", flush=True)
            continue
        # Save per-fold flat CSV
        out_csv = RESULTS_DIR / f"loo_{cfg}.csv"
        per_fold_to_csv(run_dir, out_csv)
        # Compute aggregate metrics
        m = pool_metrics(run_dir)
        summary_rows.append(m)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = RESULTS_DIR / "loo_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[runner] Summary saved to {summary_path}", flush=True)
        print(f"\n{'='*80}", flush=True)
        print("LOO Metrics — Connolly SES SC", flush=True)
        print(f"{'='*80}", flush=True)
        cols = ["config", "n_folds", "pass_rate", "pooled_auroc", "pooled_auprc",
                "pooled_accuracy", "tpr_at_0.5", "tnr_at_0.5", "bal_acc"]
        print(summary_df[cols].to_string(index=False), flush=True)

    print(f"\n[runner] Done.", flush=True)


if __name__ == "__main__":
    main()
