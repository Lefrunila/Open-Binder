#!/usr/bin/env python3
"""loo_harness.py

Leave-One-Out benchmark for OpenBinder configs.  For each of the 1129 positive antigen
stems, hold out its trio (1 positive + up to 2 negatives), train the config's
model on the remainder, and predict on the trio.

Sidecar-aware trio construction
--------------------------------
Five positives (7D30_1_cleaned, 7PH3_cleaned, 8X7N_0_cleaned, 5JMO_cleaned,
8BVT_cleaned) have at least one slot filled by a promoted orphan instead of an
antigen-derived negative.  The sidecar TSV ``data/orphan_training_assignments.tsv``
(columns: orphan_file, host_positive_stem, slot, notes) tells us which orphan
is slotted where.

Trio resolution rule:
    * Default case: trio = {positive, neg_<ANT>_*_1, neg_<ANT>_*_2} — built
      by matching the filename regex ``^neg_<STEM>__vhh_…_N.pdb`` against the
      training-set negatives.
    * Sidecar override: for each of the 5 host positives, the sidecar tells
      you which orphan fills which slot.  The other slot is filled by the
      existing antigen-derived negative found via the default regex.

We do NOT silently synthesise a negative if a slot is empty (e.g. 8BVT slot-1
has only one antigen-derived negative and slot-2 is a sidecar orphan — the
trio is size-3.  If neither slot is resolvable, the trio is size-2, which is
still a valid LOO fold.).

Outputs
-------
    models/loo/<config>/per_fold/<antigen>.json
    models/loo/<config>/pooled_metrics.json
    models/loo/<config>/pass_rate_by_antigen.csv

Usage
-----
    python scripts/v3/loo_harness.py \\
        --config configs/rf_rest.yaml \\
        --output-dir models/loo/rf_rest/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from copy import deepcopy
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datamodule import DataModule, fit_transform_esm_pca, resolve_path  # noqa: E402
from feature_combine import (  # noqa: E402
    assemble_matrix,
    esm_pca_cols,
    feature_cols_both_rf,
    feature_cols_rest,
    feature_cols_unrest,
)


NEG_PATTERN = re.compile(r"^neg_(.+?)__vhh_")


# ── Sidecar loader ───────────────────────────────────────────────────────────
def load_sidecar_assignments(tsv_path: Path) -> dict[str, dict[int, str]]:
    """Return {host_positive_stem → {slot → orphan_file}}.

    host_positive_stem is the stem (e.g. "7D30_1_cleaned") — matches the
    filename derived from ``extract_target_stem`` on orphan negatives.
    """
    if not tsv_path.exists():
        print(f"[loo] sidecar not found at {tsv_path}; assuming no orphan promotions", flush=True)
        return {}
    df = pd.read_csv(tsv_path, sep="\t")
    needed = {"orphan_file", "host_positive_stem", "slot"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"sidecar {tsv_path} missing columns: {missing}")
    out: dict[str, dict[int, str]] = {}
    for _, row in df.iterrows():
        stem = str(row["host_positive_stem"])
        slot = int(row["slot"])
        orphan = str(row["orphan_file"])
        out.setdefault(stem, {})[slot] = orphan
    return out


def resolve_trio(
    pos_stem: str,
    pos_file: str,
    antigen_derived_negs: list[str],
    sidecar: dict[int, str] | None,
) -> dict[str, list[str]]:
    """Resolve the 3-element (or 2-element) held-out trio for one positive.

    ``antigen_derived_negs`` is the ordered list of existing negatives whose
    filename matches ``neg_<STEM>__vhh_…_N.pdb`` — i.e. the "default" slot
    fillers.  ``sidecar`` is the {slot → orphan_file} mapping for this stem.

    Returns
    -------
    dict with keys 'positive', 'negatives' (list of filenames).
    """
    sidecar = sidecar or {}
    slot1 = sidecar.get(1)
    slot2 = sidecar.get(2)

    # The antigen-derived negs fill whatever slots the sidecar doesn't.
    # They usually come with _1/_2 suffixes in the filename, but we can't
    # rely on that ordering — we just fill leftover slots in the order they
    # appear.
    negs_final: list[str] = [slot1, slot2]
    remaining_antigen = [n for n in antigen_derived_negs if n not in {slot1, slot2}]

    for i in range(2):
        if negs_final[i] is None:
            if remaining_antigen:
                negs_final[i] = remaining_antigen.pop(0)
    negs_final = [n for n in negs_final if n is not None]
    return {"positive": pos_file, "negatives": negs_final}


# ── Training helpers ─────────────────────────────────────────────────────────
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


def fit_rf_fold(X_tr: np.ndarray, y_tr: np.ndarray, cfg: dict) -> Pipeline:
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", build_rf(cfg))])
    pipe.fit(X_tr, y_tr)
    return pipe


# For the MLP we delegate to mlp_train's training loop — but to keep the LOO
# loop fast for 1129 folds, we run a lightweight training loop here instead.
def fit_mlp_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    cfg: dict,
    device,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, object, int]:
    """Fit a single-branch MLP for one LOO fold with internal val-split early stopping.

    mode=both_all (etc.) goes through the flat matrix via assemble_matrix.

    Returns (pred_probs_on_X_te, fitted_state_dict, epochs_trained).
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    mlp_cfg = cfg["mlp"]
    hidden = list(mlp_cfg.get("hidden_dims", [256, 128]))
    dropout = float(mlp_cfg.get("dropout", 0.3))
    lr = float(mlp_cfg.get("lr", 1e-3))
    wd = float(mlp_cfg.get("weight_decay", 1e-4))
    bs = int(mlp_cfg.get("batch_size", 64))
    max_ep = int(mlp_cfg.get("max_epochs", 100))
    patience = int(mlp_cfg.get("patience", 10))

    # Stratified internal val split (on training set only — X_te is the held-out trio)
    idx_tr, idx_va = train_test_split(
        np.arange(len(y_tr)),
        test_size=val_frac,
        random_state=seed,
        stratify=y_tr,
    )
    X_tr_inner, y_tr_inner = X_tr[idx_tr], y_tr[idx_tr]
    X_va_inner, y_va_inner = X_tr[idx_va], y_tr[idx_va]

    scaler = StandardScaler().fit(X_tr_inner)
    X_tr_s = scaler.transform(X_tr_inner).astype(np.float32)
    X_va_s = scaler.transform(X_va_inner).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    layers: list[nn.Module] = []
    prev = X_tr_s.shape[1]
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)])
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_w = (y_tr_inner == 0).sum() / max((y_tr_inner == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))

    X_t = torch.tensor(X_tr_s, device=device)
    y_t = torch.tensor(y_tr_inner.astype(np.float32), device=device)
    X_v = torch.tensor(X_va_s, device=device)
    y_v = torch.tensor(y_va_inner.astype(np.float32), device=device)
    n = len(y_t)

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    epochs_trained = 0
    for ep in range(1, max_ep + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            logits = model(X_t[idx]).squeeze(-1)
            loss = criterion(logits, y_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_v).squeeze(-1)
            va_loss = float(criterion(va_logits, y_v).item())
        epochs_trained = ep
        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X_te_s, device=device)).squeeze(-1)).cpu().numpy()
    return probs, model.state_dict(), epochs_trained


# ── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap the number of LOO folds (debugging).")
    p.add_argument("--only-stems", type=str, default=None,
                   help="Comma-separated list of positive stems to run (skip all others).")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_fold").mkdir(parents=True, exist_ok=True)

    dm = DataModule.from_config(args.config)
    cfg = dm.config
    model_type = cfg["model_type"]
    mode = cfg["feature_mode"]
    esm_dims = int(cfg.get("esm_pca_dims", 64))

    # deferred_esm_pca=True: keep raw esm_embed_* cols; PCA is refit per fold
    # on training rows only (LOO-correct, no test-fold leakage).
    df = dm.prepare(deferred_esm_pca=True)

    # Split raw ESM block from physics columns.
    esm_raw_cols: list[str] = dm._esm_feat_cols  # 3220 raw embedding cols
    if esm_raw_cols:
        esm_raw = df[esm_raw_cols].values.astype(np.float32)  # (N, 3220)
        df_phys = df.drop(columns=esm_raw_cols)
    else:
        esm_raw = np.zeros((len(df), 0), dtype=np.float32)
        df_phys = df

    # Pre-compute canonical feature column names (for logging; values recomputed per fold).
    if mode == "rest":
        feature_cols = feature_cols_rest(esm_dims)
    elif mode == "unrest":
        feature_cols = feature_cols_unrest(esm_dims)
    elif mode in ("both", "both_delta", "both_raw", "both_all"):
        variant = "all" if mode == "both" else mode.split("_", 1)[1]
        feature_cols = feature_cols_both_rf(esm_dims, variant=variant)
    else:
        raise ValueError(f"[loo] unknown feature_mode {mode!r}")

    _pca_col_names = esm_pca_cols(esm_dims)

    y_all = dm.build_labels(df)
    files_all = df["file"].astype(str).values
    file_to_idx = {f: i for i, f in enumerate(files_all)}

    # Map positive stems and antigen-derived negatives.
    pos_mask = y_all == 1
    pos_stems: dict[str, str] = {}
    for f in files_all[pos_mask]:
        stem = f.replace(".pdb", "")
        pos_stems[stem] = f

    neg_by_target: dict[str, list[str]] = {}
    for f in files_all[~pos_mask]:
        m = NEG_PATTERN.match(f)
        if not m:
            continue
        stem = m.group(1)
        neg_by_target.setdefault(stem, []).append(f)

    # Sidecar
    sidecar_tsv = resolve_path(cfg["hold_out"]["orphan_assignments_tsv"], dm.project_root)
    sidecar = load_sidecar_assignments(sidecar_tsv)

    # Ensure sidecar orphans appear in our training cohort (they should, because
    # they live in data/negatives/ and appear in the feature CSVs).
    orphans_in_sidecar = {f for slots in sidecar.values() for f in slots.values()}
    missing_orphans = [f for f in orphans_in_sidecar if f not in file_to_idx]
    if missing_orphans:
        print(f"[loo] WARNING: sidecar lists {len(missing_orphans)} orphan(s) not in cohort: "
              f"{missing_orphans}", flush=True)

    # Build fold list (sorted for reproducibility)
    fold_stems = sorted(pos_stems.keys())
    if args.only_stems is not None:
        only_set = set(args.only_stems.split(','))
        fold_stems = [s for s in fold_stems if s in only_set]
    if args.max_folds is not None:
        fold_stems = fold_stems[:args.max_folds]

    print(f"[loo] folds={len(fold_stems)} model={model_type} mode={mode} "
          f"features={len(feature_cols)} cohort={len(y_all)} "
          f"[ESM PCA refit per fold, svd_solver=full]", flush=True)

    # Device setup for MLP
    device = None
    if model_type == "mlp":
        import torch
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[loo] MLP device: {device}", flush=True)

    results: list[dict] = []
    per_antigen_pass: list[dict] = []
    all_true: list[int] = []
    all_prob: list[float] = []

    t0 = time.time()
    for i, stem in enumerate(fold_stems, 1):
        pos_file = pos_stems[stem]
        antigen_negs = neg_by_target.get(stem, [])
        side = sidecar.get(stem)

        trio = resolve_trio(stem, pos_file, antigen_negs, side)
        held_files = [trio["positive"]] + trio["negatives"]
        held_idx = [file_to_idx[f] for f in held_files if f in file_to_idx]
        if len(held_idx) < 2:
            # Need at least the positive + 1 negative for a meaningful fold.
            results.append({"antigen": stem, "skipped": True, "reason": "trio_too_small",
                            "trio": held_files})
            continue

        held_mask = np.zeros(len(y_all), dtype=bool)
        held_mask[held_idx] = True
        tr_mask = ~held_mask

        # Per-fold ESM PCA: fit on training rows only, transform both splits.
        t_pca = time.time()
        Z_tr_esm, Z_te_esm, pca_bundle = fit_transform_esm_pca(
            esm_raw[tr_mask], esm_raw[held_mask], n_dims=esm_dims, seed=42
        )
        pca_elapsed = time.time() - t_pca

        if i == 1:
            evr5 = pca_bundle["pca"].explained_variance_ratio_[:5]
            print(f"[loo] fold-1 PCA: t={pca_elapsed:.2f}s  "
                  f"top-5 EV%={np.round(evr5 * 100, 1).tolist()}", flush=True)

        # Build fold df slices with per-fold ESM PCA columns attached.
        df_tr_fold = df_phys[tr_mask].copy().reset_index(drop=True)
        df_te_fold = df_phys[held_mask].copy().reset_index(drop=True)
        for k, col in enumerate(_pca_col_names):
            df_tr_fold[col] = Z_tr_esm[:, k]
            df_te_fold[col] = Z_te_esm[:, k]

        X_tr, _ = assemble_matrix(df_tr_fold, mode, esm_dims)
        X_te, _ = assemble_matrix(df_te_fold, mode, esm_dims)
        y_tr = y_all[tr_mask]
        y_te = y_all[held_mask]
        files_te = files_all[held_mask]

        epochs_trained: int | None = None
        fold_error: str | None = None
        if model_type == "rf":
            pipe = fit_rf_fold(X_tr, y_tr, cfg)
            probs = pipe.predict_proba(X_te)[:, 1]
        elif model_type == "mlp":
            try:
                probs, _, epochs_trained = fit_mlp_fold(X_tr, y_tr, X_te, cfg, device)
            except Exception as e:  # noqa: BLE001
                fold_error = f"{type(e).__name__}: {e}"
                print(f"[loo] fold {stem} FAILED: {fold_error}", flush=True)
                results.append({"antigen": stem, "skipped": True, "reason": "mlp_fold_error",
                                "error": fold_error, "trio": held_files})
                continue
        else:
            raise ValueError(f"unknown model_type {model_type!r}")

        # rank of the positive within the trio
        pos_i = list(files_te).index(pos_file)
        pos_prob = float(probs[pos_i])
        rank = int((probs > pos_prob).sum()) + 1
        passed = rank == 1

        per_fold = {
            "antigen": stem,
            "positive": pos_file,
            "negatives": trio["negatives"],
            "sidecar_slot": side or {},
            "pos_prob": pos_prob,
            "neg_probs": [float(p) for j, p in enumerate(probs) if j != pos_i],
            "rank": rank,
            "trio_size": len(held_idx),
            "passed": bool(passed),
        }
        if epochs_trained is not None:
            per_fold["epochs_trained"] = int(epochs_trained)
        (args.output_dir / "per_fold" / f"{stem}.json").write_text(json.dumps(per_fold, indent=2) + "\n")

        results.append(per_fold)
        per_antigen_pass.append({
            "antigen": stem,
            "pos_prob": pos_prob,
            "trio_size": len(held_idx),
            "rank": rank,
            "passed": int(passed),
        })
        all_true.extend(int(v) for v in y_te.tolist())
        all_prob.extend(float(v) for v in probs.tolist())

        if i % 50 == 0 or i == len(fold_stems):
            elapsed = time.time() - t0
            print(f"[loo] fold {i}/{len(fold_stems)} elapsed={elapsed:.1f}s", flush=True)

    # ── Pooled metrics ───────────────────────────────────────────────────
    y_pool = np.asarray(all_true, dtype=int)
    p_pool = np.asarray(all_prob, dtype=float)
    preds_pool = (p_pool >= 0.5).astype(int)

    metrics: dict[str, float | int] = {
        "n_folds": int(len(per_antigen_pass)),
        "n_pooled_samples": int(len(y_pool)),
    }
    if len(y_pool) and len(np.unique(y_pool)) > 1:
        metrics.update({
            "pooled_auroc": float(roc_auc_score(y_pool, p_pool)),
            "pooled_auprc": float(average_precision_score(y_pool, p_pool)),
            "pooled_accuracy": float(accuracy_score(y_pool, preds_pool)),
        })
    if per_antigen_pass:
        pass_rate = sum(r["passed"] for r in per_antigen_pass) / len(per_antigen_pass)
        metrics["pass_rate"] = float(pass_rate)

    (args.output_dir / "pooled_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(per_antigen_pass).to_csv(args.output_dir / "pass_rate_by_antigen.csv", index=False)
    print(f"[loo] pooled metrics: {json.dumps(metrics, indent=2)}", flush=True)
    print(f"[loo] artifacts written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
