#!/usr/bin/env python3
"""loo_compare.py — aggregate LOO per-fold outputs into a comparison table.

Reads models/loo/<run>/per_fold/*.json for a list of run directories, pools
the predictions, and prints (a) pooled AUROC / AUPRC / pass_rate / TPR@0.5 /
TNR@0.5 / BalAcc and (b) borderline-failure breakdown (positives missed by
<0.1 below threshold; negatives confidently wrong with prob > 0.7).

No side effects — read-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def pool_run(run_dir: Path) -> dict:
    per_fold_dir = run_dir / "per_fold"
    files = sorted(per_fold_dir.glob("*.json"))
    y_true: list[int] = []
    y_prob: list[float] = []
    rows: list[dict] = []
    skipped: list[dict] = []
    n_pos = n_neg = 0
    pos_probs: list[float] = []
    neg_probs: list[float] = []
    passes = 0
    trio_sizes: list[int] = []
    epochs_trained: list[int] = []
    for f in files:
        d = json.loads(f.read_text())
        if d.get("skipped"):
            skipped.append(d)
            continue
        pos_probs.append(d["pos_prob"])
        y_true.append(1)
        y_prob.append(d["pos_prob"])
        n_pos += 1
        for p in d["neg_probs"]:
            neg_probs.append(p)
            y_true.append(0)
            y_prob.append(p)
            n_neg += 1
        passes += int(d["passed"])
        trio_sizes.append(d["trio_size"])
        if "epochs_trained" in d:
            epochs_trained.append(d["epochs_trained"])
        rows.append(d)

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

    # borderline failure breakdown
    pos_probs_arr = np.asarray(pos_probs)
    neg_probs_arr = np.asarray(neg_probs)
    pos_fail_mask = pos_probs_arr < 0.5
    neg_fail_mask = neg_probs_arr >= 0.5
    n_pos_fail = int(pos_fail_mask.sum())
    n_neg_fail = int(neg_fail_mask.sum())
    borderline_pos = int(((pos_probs_arr >= 0.4) & pos_fail_mask).sum())  # within 0.1 of threshold
    confident_neg = int((neg_probs_arr > 0.7).sum())
    borderline_pos_frac = borderline_pos / max(n_pos_fail, 1)
    confident_neg_frac = confident_neg / max(n_neg_fail, 1)

    res = {
        "run": run_dir.name,
        "n_folds_used": len(rows),
        "n_skipped": len(skipped),
        "skipped_reasons": [s.get("reason") for s in skipped],
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pass_rate": passes / max(len(rows), 1),
        "pooled_auroc": float(roc_auc_score(y_true, y_prob)),
        "pooled_auprc": float(average_precision_score(y_true, y_prob)),
        "pooled_acc": float((preds == y_true).mean()),
        "tpr_at_0.5": float(tpr),
        "tnr_at_0.5": float(tnr),
        "bal_acc": float(bal_acc),
        "pos_mean_prob": float(np.mean(pos_probs)),
        "neg_mean_prob": float(np.mean(neg_probs)),
        "n_pos_fail": n_pos_fail,
        "n_neg_fail": n_neg_fail,
        "borderline_pos_fail": borderline_pos,
        "borderline_pos_fail_frac": float(borderline_pos_frac),
        "confident_neg_fail": confident_neg,
        "confident_neg_fail_frac": float(confident_neg_frac),
    }
    if epochs_trained:
        res["epochs_median"] = int(np.median(epochs_trained))
        res["epochs_mean"] = float(np.mean(epochs_trained))
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    rows = [pool_run(r) for r in args.runs]
    df = pd.DataFrame(rows)
    cols = [
        "run", "n_folds_used", "n_skipped", "pass_rate",
        "pooled_auroc", "pooled_auprc", "pooled_acc",
        "tpr_at_0.5", "tnr_at_0.5", "bal_acc",
        "pos_mean_prob", "neg_mean_prob",
        "n_pos_fail", "borderline_pos_fail", "borderline_pos_fail_frac",
        "n_neg_fail", "confident_neg_fail", "confident_neg_fail_frac",
    ]
    if "epochs_median" in df.columns:
        cols += ["epochs_median", "epochs_mean"]
    print(df[cols].to_string(index=False))
    if args.out is not None:
        df.to_csv(args.out, index=False)
        print(f"\n[compare] wrote {args.out}")


if __name__ == "__main__":
    main()
