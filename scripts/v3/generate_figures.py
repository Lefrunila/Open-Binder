#!/usr/bin/env python3
"""generate_figures.py

Generates publication-quality figures for the OpenBinder paper.

Run from the Open-Binder repo root:
    python scripts/v3/generate_figures.py

Figures are saved to:
    Open-Binder/docs/figures/
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Open-Binder/
# LOO results: per-fold StandardScaler + PCA refit on training rows only,
# 1129 folds, open-source Connolly SES sc.  No global-cohort leakage.
LOO_BASE = PROJECT_ROOT / "models" / "loo_results"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints" / "rf_both_all"
OUT_DIR = PROJECT_ROOT / "docs" / "figures"
COPY_DIR = PROJECT_ROOT / "figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
COPY_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
BLUE   = "#2166ac"   # positives / MLP
RED    = "#d6604d"   # negatives / RF
GREEN  = "#4dac26"   # ensemble
LGRAY  = "#cccccc"   # gridlines

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.grid": True,
    "grid.color": LGRAY,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ── LOO directory map ────────────────────────────────────────────────────────
def find_loo_dir(prefix: str) -> Path:
    # Connolly LOO dirs are named exactly (no timestamp suffix)
    exact = LOO_BASE / prefix
    if exact.is_dir():
        return exact
    # Fallback: glob for timestamped dirs (legacy Rosetta layout)
    matches = sorted(LOO_BASE.glob(f"{prefix}_*"))
    if not matches:
        raise FileNotFoundError(
            f"No LOO dir named '{prefix}' or matching {prefix}_* in {LOO_BASE}"
        )
    return matches[-1]  # most recent


def load_per_fold(prefix: str) -> tuple[list[float], list[float]]:
    """Return (pos_probs, neg_probs) lists from per_fold JSONs."""
    loo_dir = find_loo_dir(prefix)
    pos_probs: list[float] = []
    neg_probs: list[float] = []
    for jf in sorted((loo_dir / "per_fold").glob("*.json")):
        data = json.loads(jf.read_text())
        pos_probs.append(data["pos_prob"])
        neg_probs.extend(data["neg_probs"])
    return pos_probs, neg_probs


def save_fig(fig: plt.Figure, name: str) -> None:
    pdf = OUT_DIR / f"{name}.pdf"
    png = OUT_DIR / f"{name}.png"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    if COPY_DIR != OUT_DIR:
        shutil.copy2(pdf, COPY_DIR / f"{name}.pdf")
        shutil.copy2(png, COPY_DIR / f"{name}.png")
    print(f"  Saved {pdf}  ({pdf.stat().st_size // 1024} KB)")
    print(f"  Saved {png}  ({png.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 (manuscript): Score distributions — RF vs MLP
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4_score_distributions() -> None:
    print("Generating Figure 4 (score distributions) …")
    rf_pos,  rf_neg  = load_per_fold("rf_both_all")
    mlp_pos, mlp_neg = load_per_fold("mlp_both_all")

    def panel_stats(pos, neg):
        tpr = sum(p >= 0.5 for p in pos) / len(pos)
        tnr = sum(p <  0.5 for p in neg) / len(neg)
        return tpr, tnr

    rf_tpr,  rf_tnr  = panel_stats(rf_pos,  rf_neg)
    mlp_tpr, mlp_tnr = panel_stats(mlp_pos, mlp_neg)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    fig.suptitle(
        "Score distributions on leave-one-out pool (N=3,387)",
        fontsize=12, fontweight="bold", y=1.01,
    )

    bins = np.linspace(0, 1, 41)

    for ax, pos, neg, model_name, tpr, tnr, xlabel in [
        (axes[0], rf_pos,  rf_neg,  "RF",  rf_tpr,  rf_tnr,
         "RF binding probability"),
        (axes[1], mlp_pos, mlp_neg, "MLP", mlp_tpr, mlp_tnr,
         "MLP binding probability"),
    ]:
        ax.hist(neg, bins=bins, color=RED,  alpha=0.6, label="True negatives")
        ax.hist(pos, bins=bins, color=BLUE, alpha=0.6, label="True positives")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2,
                   label="Threshold (0.5)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Panel {'A' if model_name == 'RF' else 'B'}: {model_name}")
        ax.legend(fontsize=9, framealpha=0.8)
        textstr = f"TPR = {tpr:.3f}\nTNR = {tnr:.3f}"
        ax.text(0.97, 0.97, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor=LGRAY, alpha=0.9))

    fig.tight_layout()
    save_fig(fig, "fig4")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 (manuscript): RF–MLP complementarity scatter
# ─────────────────────────────────────────────────────────────────────────────
def make_fig5_rf_mlp_scatter() -> None:
    print("Generating Figure 5 (RF–MLP scatter) …")
    # Build paired (rf_score, mlp_score, label) per structure
    rf_dir  = find_loo_dir("rf_both_all")  / "per_fold"
    mlp_dir = find_loo_dir("mlp_both_all") / "per_fold"

    rf_scores_pos, rf_scores_neg = [], []
    mlp_scores_pos, mlp_scores_neg = [], []

    for jf in sorted(rf_dir.glob("*.json")):
        rfdata  = json.loads(jf.read_text())
        mlpfile = mlp_dir / jf.name
        if not mlpfile.exists():
            continue
        mlpdata = json.loads(mlpfile.read_text())

        # Positive: single pair
        rf_scores_pos.append(rfdata["pos_prob"])
        mlp_scores_pos.append(mlpdata["pos_prob"])

        # Negatives: matched by position
        for r, m in zip(rfdata["neg_probs"], mlpdata["neg_probs"]):
            rf_scores_neg.append(r)
            mlp_scores_neg.append(m)

    rf_pos_arr  = np.array(rf_scores_pos)
    mlp_pos_arr = np.array(mlp_scores_pos)
    rf_neg_arr  = np.array(rf_scores_neg)
    mlp_neg_arr = np.array(mlp_scores_neg)

    n_pos = len(rf_pos_arr)
    n_neg = len(rf_neg_arr)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.scatter(rf_neg_arr,  mlp_neg_arr,  color=RED,  alpha=0.15, s=6,
               label=f"True negatives (n={n_neg:,})", zorder=2)
    ax.scatter(rf_pos_arr,  mlp_pos_arr,  color=BLUE, alpha=0.4, s=15,
               label=f"True positives (n={n_pos:,})", zorder=3)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0)

    # Quadrant counts on the full dataset (all positives + all negatives).
    q_tr   = int(np.sum((rf_pos_arr >= 0.5) & (mlp_pos_arr >= 0.5)))
    q_tr_n = int(np.sum((rf_neg_arr >= 0.5) & (mlp_neg_arr >= 0.5)))
    q_bl   = int(np.sum((rf_pos_arr <  0.5) & (mlp_pos_arr <  0.5)))
    q_bl_n = int(np.sum((rf_neg_arr <  0.5) & (mlp_neg_arr <  0.5)))
    q_br   = int(np.sum((rf_pos_arr >= 0.5) & (mlp_pos_arr <  0.5)))
    q_br_n = int(np.sum((rf_neg_arr >= 0.5) & (mlp_neg_arr <  0.5)))
    q_tl   = int(np.sum((rf_pos_arr <  0.5) & (mlp_pos_arr >= 0.5)))
    q_tl_n = int(np.sum((rf_neg_arr <  0.5) & (mlp_neg_arr >= 0.5)))

    label_kw = dict(fontsize=8.5, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor=LGRAY,
                              alpha=0.85, boxstyle="round,pad=0.3"))
    ax.text(0.75, 0.75,
            f"Both agree:\nbinder\n(pos: {q_tr}, neg: {q_tr_n})",
            transform=ax.transAxes, **label_kw)
    ax.text(0.25, 0.25,
            f"Both agree:\nnon-binder\n(pos: {q_bl}, neg: {q_bl_n})",
            transform=ax.transAxes, **label_kw)
    ax.text(0.75, 0.25,
            f"RF only\n(pos: {q_br}, neg: {q_br_n})",
            transform=ax.transAxes, **label_kw)
    ax.text(0.25, 0.75,
            f"MLP only\n(pos: {q_tl}, neg: {q_tl_n})",
            transform=ax.transAxes, **label_kw)

    ax.set_xlabel("RF binding probability (rf_both_all)")
    ax.set_ylabel("MLP binding probability (mlp_both_all)")
    ax.set_title("RF–MLP score concordance on LOO pool")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig5")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 (manuscript): Feature importance (rf_both_all)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig6_feature_importance() -> None:
    print("Generating Figure 6 (feature importances) …")
    feat_csv = CHECKPOINT_DIR / "feature_importances.csv"
    if not feat_csv.exists():
        print(f"  WARNING: {feat_csv} not found — skipping Fig 6")
        return

    df = pd.read_csv(feat_csv)
    df = df.sort_values("importance", ascending=False).head(20)
    df = df.iloc[::-1]  # reverse for horizontal bar (top at top)

    # Classify feature families
    BLUE_DARK  = "#2166ac"
    BLUE_MED   = "#74add1"
    BLUE_LIGHT = "#abd9e9"
    RED_DARK   = "#d73027"
    RED_MED    = "#f46d43"
    GREEN_COL  = "#4dac26"

    def classify(feat: str) -> str:
        if feat.startswith("esm_pca"):
            return "ESM PCA"
        if feat in ("n_salt_bridges_int", "n_aromatic_stacking_int",
                    "n_hydrophobic_int", "n_repulsive_int"):
            return "COCaDA rest"
        if feat.endswith("__unrest"):
            base = feat[:-len("__unrest")]
            if base in ("n_salt_bridges_int", "n_aromatic_stacking_int",
                        "n_hydrophobic_int", "n_repulsive_int"):
                return "COCaDA unrest"
            return "OpenMM unrest"
        if feat.endswith("__delta"):
            return "OpenMM delta"
        # check COCaDA base
        cocada_bases = {"n_salt_bridges_int", "n_aromatic_stacking_int",
                        "n_hydrophobic_int", "n_repulsive_int"}
        if feat in cocada_bases:
            return "COCaDA rest"
        return "OpenMM rest"

    family_color = {
        "OpenMM rest":   BLUE_DARK,
        "OpenMM unrest": BLUE_MED,
        "OpenMM delta":  BLUE_LIGHT,
        "COCaDA rest":   RED_DARK,
        "COCaDA unrest": RED_MED,
        "ESM PCA":       GREEN_COL,
    }

    df["family"] = df["feature"].apply(classify)
    df["color"]  = df["family"].map(family_color)

    # Build readable labels
    def readable_label(feat: str) -> str:
        if feat.endswith("__unrest"):
            return feat[:-len("__unrest")] + " (unrest)"
        if feat.endswith("__delta"):
            return feat[:-len("__delta")] + " (Δ)"
        return feat

    df["label"] = df["feature"].apply(readable_label)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    bars = ax.barh(df["label"], df["importance"], color=df["color"].tolist(),
                   edgecolor="none", height=0.7)

    ax.set_xlabel("Gini feature importance")
    ax.set_title("Top 20 feature importances — rf_both_all (153 features)")
    ax.set_xlim(0, df["importance"].max() * 1.18)
    ax.tick_params(axis="y", labelsize=9)

    # Legend
    patches = [mpatches.Patch(color=c, label=lbl)
               for lbl, c in family_color.items()]
    ax.legend(handles=patches, fontsize=8.5, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig6")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 (manuscript): LOO ablation — AUROC/AUPRC by feature mode
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2_ablation() -> None:
    print("Generating Figure 2 (ablation) …")

    models_ordered = [
        ("rf_rest",       "rf_rest"),
        ("rf_unrest",     "rf_unrest"),
        ("rf_both_raw",   "rf_both_raw"),
        ("rf_both_delta", "rf_both_delta"),
        ("rf_both_all",   "rf_both_all"),
        ("mlp_both_all",  "mlp_both_all"),
    ]

    labels, auroc_vals, auprc_vals = [], [], []
    for label, prefix in models_ordered:
        loo_dir = find_loo_dir(prefix)
        metrics = json.loads((loo_dir / "pooled_metrics.json").read_text())
        labels.append(label)
        auroc_vals.append(metrics["pooled_auroc"])
        auprc_vals.append(metrics["pooled_auprc"])

    # Sort by AUROC ascending (worst to best on Y from bottom)
    order = np.argsort(auroc_vals)
    labels     = [labels[i]     for i in order]
    auroc_vals = [auroc_vals[i] for i in order]
    auprc_vals = [auprc_vals[i] for i in order]

    # Colors: RF shades light→dark, MLP green
    rf_shades = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]
    model_colors_auroc = []
    model_colors_auprc = []
    rf_idx = 0
    for lbl in labels:
        if lbl.startswith("mlp"):
            model_colors_auroc.append(GREEN)
            model_colors_auprc.append("#74c476")
        else:
            model_colors_auroc.append(rf_shades[rf_idx])
            model_colors_auprc.append(rf_shades[rf_idx])
            rf_idx += 1

    y = np.arange(len(labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_auroc = ax.barh(y + height/2, auroc_vals, height=height,
                         color=model_colors_auroc, edgecolor="white",
                         linewidth=0.5, label="AUROC")
    bars_auprc = ax.barh(y - height/2, auprc_vals, height=height,
                         color=model_colors_auprc, edgecolor="white",
                         linewidth=0.5, alpha=0.75, label="AUPRC",
                         hatch="///")

    ax.axvline(0.9, color="black", linestyle="--", linewidth=1.0,
               alpha=0.6, label="Reference (0.9)")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_title("Leave-one-out performance across feature configurations")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)

    # Annotate values
    for i, (au, ap) in enumerate(zip(auroc_vals, auprc_vals)):
        ax.text(au + 0.001, i + height/2, f"{au:.4f}",
                va="center", fontsize=7.5, color="black")
        ax.text(ap + 0.001, i - height/2, f"{ap:.4f}",
                va="center", fontsize=7.5, color="black")

    fig.tight_layout()
    save_fig(fig, "fig2")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 (manuscript): Precision-Recall curves — RF and MLP
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3_pr_curves() -> None:
    print("Generating Figure 3 (PR curves) …")
    from sklearn.metrics import precision_recall_curve, auc

    rf_pos,  rf_neg  = load_per_fold("rf_both_all")
    mlp_pos, mlp_neg = load_per_fold("mlp_both_all")

    def pr_data(pos, neg):
        y_true = [1] * len(pos) + [0] * len(neg)
        y_score = pos + neg
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        return prec, rec, auc(rec, prec)

    rf_prec,  rf_rec,  rf_auprc  = pr_data(rf_pos,  rf_neg)
    mlp_prec, mlp_rec, mlp_auprc = pr_data(mlp_pos, mlp_neg)

    # Baseline: fraction of positives
    baseline = len(rf_pos) / (len(rf_pos) + len(rf_neg))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.plot(rf_rec,  rf_prec,  color=RED,  linewidth=1.8,
            label=f"RF (rf_both_all)  AUPRC = {rf_auprc:.4f}")
    ax.plot(mlp_rec, mlp_prec, color=BLUE, linewidth=1.8,
            label=f"MLP (mlp_both_all)  AUPRC = {mlp_auprc:.4f}")
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1.0,
               label=f"Random classifier (prevalence = {baseline:.2f})")

    ax.set_xlabel("Recall (True Positive Rate)")
    ax.set_ylabel("Precision (Positive Predictive Value)")
    ax.set_title("Precision-Recall curves — LOO pool (N = 3,387)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig3")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary: Calibration (reliability) diagrams — RF and MLP
# Not in the manuscript but kept as a supplementary figure (figS_calibration).
# ─────────────────────────────────────────────────────────────────────────────
def make_fig_calibration() -> None:
    print("Generating supplementary calibration figure …")
    from sklearn.calibration import calibration_curve

    rf_pos,  rf_neg  = load_per_fold("rf_both_all")
    mlp_pos, mlp_neg = load_per_fold("mlp_both_all")

    def calib_data(pos, neg, n_bins=10):
        y_true  = np.array([1] * len(pos) + [0] * len(neg))
        y_score = np.array(pos + neg)
        frac_pos, mean_pred = calibration_curve(y_true, y_score,
                                                n_bins=n_bins, strategy="uniform")
        return frac_pos, mean_pred

    rf_frac,  rf_mean  = calib_data(rf_pos,  rf_neg)
    mlp_frac, mlp_mean = calib_data(mlp_pos, mlp_neg)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    fig.suptitle("Calibration (reliability) diagrams — LOO pool", fontsize=12,
                 fontweight="bold", y=1.01)

    for ax, frac, mean, color, title in [
        (axes[0], rf_frac,  rf_mean,  RED,  "Panel A: RF (rf_both_all)"),
        (axes[1], mlp_frac, mlp_mean, BLUE, "Panel B: MLP (mlp_both_all)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect calibration")
        ax.plot(mean, frac, "o-", color=color, linewidth=1.8, markersize=6,
                label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of true positives")
        ax.set_title(title)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "figS_calibration")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 (manuscript): sc_connolly vs Rosetta SC validation
# (pre-computed by sc_connolly_validate.py and saved as sc_validation.{pdf,png};
#  this function copies that artifact to the canonical fig1.{pdf,png} names.)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1_sc_validation() -> None:
    print("Registering Figure 1 (sc_validation) …")
    for ext in ("pdf", "png"):
        src = OUT_DIR / f"sc_validation.{ext}"
        dst = OUT_DIR / f"fig1.{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            if COPY_DIR != OUT_DIR:
                shutil.copy2(src, COPY_DIR / f"fig1.{ext}")
            print(f"  Copied {src} → {dst}  ({dst.stat().st_size // 1024} KB)")
        else:
            print(f"  WARNING: {src} not found — skipping Fig 1.{ext}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Manuscript-canonical figures (numbering follows OpenBinder paper):
    #   Fig 1 — SC validation (Connolly vs Rosetta)
    #   Fig 2 — LOO AUROC/AUPRC ablation across feature configurations
    #   Fig 3 — Precision–recall curves (RF and MLP)
    #   Fig 4 — Score distributions (RF vs MLP histograms)
    #   Fig 5 — RF–MLP concordance scatter
    #   Fig 6 — Top-20 Gini feature importances (rf_both_all)
    # Supplementary:
    #   figS_calibration — reliability diagrams (RF, MLP)
    make_fig1_sc_validation()
    make_fig2_ablation()
    make_fig3_pr_curves()
    make_fig4_score_distributions()
    make_fig5_rf_mlp_scatter()
    make_fig6_feature_importance()
    make_fig_calibration()

    print("\nVerifying output files:")
    for n in range(1, 7):
        for ext in ("pdf", "png"):
            p = OUT_DIR / f"fig{n}.{ext}"
            status = f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING"
            print(f"  fig{n}.{ext}: {status}")
    for ext in ("pdf", "png"):
        p = OUT_DIR / f"figS_calibration.{ext}"
        status = f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING"
        print(f"  figS_calibration.{ext}: {status}")

    print("\nDone.")
