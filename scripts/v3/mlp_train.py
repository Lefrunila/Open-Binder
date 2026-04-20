#!/usr/bin/env python3
"""mlp_train.py

Train an OpenBinder MLP (PyTorch) on the full (non-LOO) training set for one config.

Three flavours, controlled by ``feature_mode``:

    rest   → single-branch MLP over 27 OpenMM + 4 COCaDA + ESM-PCA
    unrest → single-branch MLP over 27 OpenMM (unrest)
    both   → TWO-branch MLP: rest (27) and unrest (27) each → 128d encoder,
             late concat with COCaDA + ESM-PCA → classifier head

Writes to ``--output-dir``:
    model.pt             — best checkpoint (by validation loss)
    metrics.json         — AUROC / AUPRC / accuracy on stratified hold-out
    predictions.csv      — file, label, pred_prob (per-sample)
    training_curve.csv   — per-epoch train/val loss + metrics
    meta.json            — config snapshot + cohort sizes

Usage
-----
    python scripts/v3/mlp_train.py \\
        --config configs/mlp_both_all.yaml \\
        --output-dir models/runs/mlp_both_all_<timestamp>/
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from datamodule import DataModule  # noqa: E402
from feature_combine import (  # noqa: E402
    COCADA_COLS_4,
    OPENMM_COLS_27,
    assemble_matrix,
    esm_pca_cols,
    feature_cols_rest,
    feature_cols_unrest,
    unrest_suffix,
)


# ── Model definitions ────────────────────────────────────────────────────────
class SingleBranchMLP(nn.Module):
    """Simple fully-connected MLP for rest or unrest mode."""

    def __init__(self, n_features: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x).squeeze(-1)


class TwoBranchMLP(nn.Module):
    """Two-branch MLP for mode=both.

    Each of the 27-dim OpenMM vectors (rest, unrest) is fed through an
    encoder to produce a 128-d representation; those are concatenated with
    the 4 COCaDA features and the ESM-PCA vector, then passed through a
    classifier head.  The two encoders have independent weights (a
    shared-encoder variant is a straightforward swap but we keep them
    separate so each branch can specialise to its relaxation regime).
    """

    def __init__(
        self,
        n_rest: int,
        n_unrest: int,
        n_extras: int,
        hidden_dims: list[int],
        dropout: float,
        branch_out: int = 128,
    ):
        super().__init__()
        self.rest_enc = self._encoder(n_rest, branch_out, dropout)
        self.unrest_enc = self._encoder(n_unrest, branch_out, dropout)
        head_in = 2 * branch_out + n_extras
        layers: list[nn.Module] = []
        prev = head_in
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    @staticmethod
    def _encoder(n_in: int, n_out: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_out, n_out),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
        )

    def forward(self, rest: torch.Tensor, unrest: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h_rest = self.rest_enc(rest)
        h_unrest = self.unrest_enc(unrest)
        h = torch.cat([h_rest, h_unrest, extras], dim=1)
        return self.head(h).squeeze(-1)


# ── Helpers ──────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None,
                   help="Override device. Defaults to cuda if available else cpu.")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Override cfg['mlp']['max_epochs'] for this run.")
    return p.parse_args()


def extract_blocks(df: pd.DataFrame, mode: str, esm_dims: int) -> dict[str, np.ndarray]:
    """Return per-branch feature blocks as float32 arrays."""
    blocks: dict[str, np.ndarray] = {}
    if mode == "rest":
        blocks["x"] = df[feature_cols_rest(esm_dims)].values.astype(np.float32)
    elif mode == "unrest":
        blocks["x"] = df[feature_cols_unrest(esm_dims)].values.astype(np.float32)
    elif mode == "both":
        blocks["rest"] = df[OPENMM_COLS_27].values.astype(np.float32)
        blocks["unrest"] = df[[unrest_suffix(c) for c in OPENMM_COLS_27]].values.astype(np.float32)
        extra_cols = list(COCADA_COLS_4) + esm_pca_cols(esm_dims)
        blocks["extras"] = df[extra_cols].values.astype(np.float32)
    elif mode in ("both_all", "both_delta", "both_raw"):
        X, _ = assemble_matrix(df, mode, esm_dims)
        blocks["x"] = X
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return blocks


def fit_scalers(blocks: dict[str, np.ndarray]) -> dict[str, StandardScaler]:
    return {k: StandardScaler().fit(v) for k, v in blocks.items()}


def apply_scalers(blocks: dict[str, np.ndarray], scalers: dict[str, StandardScaler]) -> dict[str, np.ndarray]:
    return {k: scalers[k].transform(v).astype(np.float32) for k, v in blocks.items()}


def train_model(
    model: nn.Module,
    mode: str,
    tr_blocks: dict[str, torch.Tensor],
    y_tr: torch.Tensor,
    va_blocks: dict[str, torch.Tensor],
    y_va: torch.Tensor,
    cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, list[dict]]:
    mlp_cfg = cfg["mlp"]
    lr = float(mlp_cfg.get("lr", 1e-3))
    wd = float(mlp_cfg.get("weight_decay", 1e-4))
    bs = int(mlp_cfg.get("batch_size", 64))
    max_ep = int(mlp_cfg.get("max_epochs", 100))
    patience = int(mlp_cfg.get("patience", 10))

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum().item(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    n_tr = y_tr.shape[0]
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    curve: list[dict] = []

    for epoch in range(1, max_ep + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        tr_loss_sum = 0.0
        n_batches = 0
        for start in range(0, n_tr, bs):
            idx = perm[start:start + bs]
            if mode == "both":
                logits = model(tr_blocks["rest"][idx], tr_blocks["unrest"][idx], tr_blocks["extras"][idx])
            else:
                logits = model(tr_blocks["x"][idx])
            loss = criterion(logits, y_tr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_loss_sum += float(loss.item())
            n_batches += 1

        # validation
        model.eval()
        with torch.no_grad():
            if mode == "both":
                va_logits = model(va_blocks["rest"], va_blocks["unrest"], va_blocks["extras"])
            else:
                va_logits = model(va_blocks["x"])
            va_loss = float(criterion(va_logits, y_va).item())
            va_probs = torch.sigmoid(va_logits).cpu().numpy()
            va_labels = y_va.cpu().numpy().astype(int)
            try:
                auroc = float(roc_auc_score(va_labels, va_probs))
            except ValueError:
                auroc = float("nan")

        tr_loss = tr_loss_sum / max(n_batches, 1)
        curve.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_auroc": auroc})

        if va_loss < best_val - 1e-6:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"[mlp_train] early stop at epoch {epoch} (best val_loss={best_val:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dm = DataModule.from_config(args.config)
    cfg = dm.config
    if cfg["model_type"] != "mlp":
        raise SystemExit(f"mlp_train expects model_type=mlp, got {cfg['model_type']!r}")
    if args.max_epochs is not None:
        cfg.setdefault("mlp", {})["max_epochs"] = int(args.max_epochs)
        print(f"[mlp_train] override max_epochs={args.max_epochs} from CLI", flush=True)

    df = dm.prepare()
    esm_dims = int(cfg.get("esm_pca_dims", 64))
    mode = cfg["feature_mode"]

    blocks = extract_blocks(df, mode, esm_dims)
    y = dm.build_labels(df).astype(np.float32)
    files = df["file"].astype(str).values

    # Stratified split
    idx_tr, idx_va = train_test_split(
        np.arange(len(y)),
        test_size=args.val_frac,
        random_state=args.seed,
        stratify=y,
    )
    tr_raw = {k: v[idx_tr] for k, v in blocks.items()}
    va_raw = {k: v[idx_va] for k, v in blocks.items()}

    # Per-branch scalers fit on training split only
    scalers = fit_scalers(tr_raw)
    tr_scaled = apply_scalers(tr_raw, scalers)
    va_scaled = apply_scalers(va_raw, scalers)

    # To torch
    def to_t(d: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in d.items()}

    tr_t = to_t(tr_scaled)
    va_t = to_t(va_scaled)
    y_tr_t = torch.tensor(y[idx_tr], dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y[idx_va], dtype=torch.float32, device=device)

    # Build model
    mlp_cfg = cfg["mlp"]
    hidden = list(mlp_cfg.get("hidden_dims", [256, 128]))
    dropout = float(mlp_cfg.get("dropout", 0.3))

    if mode == "both":
        n_rest = tr_t["rest"].shape[1]
        n_unrest = tr_t["unrest"].shape[1]
        n_extras = tr_t["extras"].shape[1]
        model = TwoBranchMLP(n_rest, n_unrest, n_extras, hidden, dropout).to(device)
    else:
        n_in = tr_t["x"].shape[1]
        model = SingleBranchMLP(n_in, hidden, dropout).to(device)

    print(f"[mlp_train] device={device} mode={mode} train_n={len(idx_tr)} val_n={len(idx_va)}", flush=True)
    model, curve = train_model(model, mode, tr_t, y_tr_t, va_t, y_va_t, cfg, device)

    # Metrics on val
    model.eval()
    with torch.no_grad():
        if mode == "both":
            probs_va = torch.sigmoid(model(va_t["rest"], va_t["unrest"], va_t["extras"])).cpu().numpy()
        else:
            probs_va = torch.sigmoid(model(va_t["x"])).cpu().numpy()
    preds_va = (probs_va >= 0.5).astype(int)
    y_va_np = y[idx_va].astype(int)

    metrics = {
        "n_train": int(len(idx_tr)),
        "n_val": int(len(idx_va)),
        "auroc_val": float(roc_auc_score(y_va_np, probs_va)),
        "auprc_val": float(average_precision_score(y_va_np, probs_va)),
        "accuracy_val": float(accuracy_score(y_va_np, preds_va)),
        "n_epochs_trained": len(curve),
    }

    # Predictions on full set (using the model fit on tr+val scalers; for the MLP we
    # do not refit on the full cohort — the early-stopped checkpoint is the artifact.)
    full_scaled = apply_scalers(blocks, scalers)
    full_t = to_t(full_scaled)
    with torch.no_grad():
        if mode == "both":
            probs_all = torch.sigmoid(model(full_t["rest"], full_t["unrest"], full_t["extras"])).cpu().numpy()
        else:
            probs_all = torch.sigmoid(model(full_t["x"])).cpu().numpy()

    torch.save({
        "state_dict": model.state_dict(),
        "mode": mode,
        "hidden_dims": hidden,
        "dropout": dropout,
        "scalers": {k: {"mean": v.mean_.tolist(), "scale": v.scale_.tolist()} for k, v in scalers.items()},
    }, args.output_dir / "model.pt")

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    pd.DataFrame({
        "file": files,
        "label": y.astype(int),
        "pred_prob": probs_all,
    }).to_csv(args.output_dir / "predictions.csv", index=False)

    pd.DataFrame(curve).to_csv(args.output_dir / "training_curve.csv", index=False)

    meta = {
        "config_name": cfg["config_name"],
        "model_type": cfg["model_type"],
        "feature_mode": mode,
        "esm_pca_dims": esm_dims,
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "hidden_dims": hidden,
        "dropout": dropout,
        "device": str(device),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[mlp_train] metrics: {json.dumps(metrics, indent=2)}", flush=True)
    print(f"[mlp_train] artifacts written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
