#!/usr/bin/env python3
"""datamodule.py

Feature loader / cohort builder for OpenBinder training.

Responsibilities
----------------
1. Read YAML config (paths resolved relative to project root).
2. Load every feature source listed in the config.
3. Inner-join on ``file`` to produce an **intersection cohort** — each sample
   has rows in every required feature source (per ``feature_mode``).
4. Apply hold-out: strip the 30 orphans listed in ``held_out_orphans.tsv``
   from the training set (they're reserved for OOD eval).
5. Drop rows where any OpenMM extractor produced ``status == "ERROR"``.
6. Fit a PCA on the ESM-PPI block, project to ``esm_pca_dims`` (default 64).
7. Return a flat DataFrame ready for either RF or MLP training, plus the
   feature column ordering.

The class is config-driven so the trainer entry points (``rf_train`` /
``mlp_train``) and the LOO harness all see identical cohorts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from feature_combine import (
    COCADA_COLS_4,
    OPENMM_COLS_27,
    esm_pca_cols,
    unrest_suffix,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


# ── Helpers ──────────────────────────────────────────────────────────────────
NEG_PATTERN = re.compile(r"neg_(.+?)__vhh_")


def extract_target_stem(filename: str) -> str | None:
    """``neg_{STEM}__vhh_…`` → ``STEM_cleaned`` (matches positive stem).

    The convention in this project: a positive is named ``<STEM>.pdb`` with
    ``STEM`` ending in ``_cleaned``; its matching negatives are
    ``neg_<STEM>__vhh_<DONOR>_<N>.pdb``.  Note ``STEM`` already contains
    ``_cleaned``, so no extra suffix is needed.
    """
    m = NEG_PATTERN.match(filename)
    return m.group(1) if m else None


def resolve_path(p: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve a (possibly relative) path against the project root."""
    p = Path(p)
    return p if p.is_absolute() else project_root / p


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config and return it as a plain dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── DataModule ───────────────────────────────────────────────────────────────
@dataclass
class DataModule:
    """Config-driven loader for OpenBinder feature matrices.

    Usage
    -----
    >>> dm = DataModule.from_config("configs/rf_rest.yaml")
    >>> df = dm.load_features()
    >>> X, cols = dm.build_matrix(df)
    >>> y = dm.build_labels(df)
    """

    config: dict[str, Any]
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    # populated lazily
    _pca: PCA | None = None
    _esm_scaler_mean: np.ndarray | None = None
    _esm_scaler_std: np.ndarray | None = None

    # ── ctors ────────────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str | Path, project_root: Path | None = None) -> "DataModule":
        cfg = load_config(config_path)
        if project_root is None:
            project_root = PROJECT_ROOT
        return cls(config=cfg, project_root=project_root)

    # ── Loading / merging ────────────────────────────────────────────────
    def _read_csv(self, key: str) -> pd.DataFrame:
        """Read a feature CSV, tolerating the handful of legacy rows with a
        trailing duplicate column (Issue #1 — a stale ``hbond_density`` copy
        emitted before the CSV header was patched).  We use
        ``on_bad_lines='skip'`` plus a warning so those rows are silently
        dropped rather than aborting the whole load.
        """
        src = self.config["feature_sources"][key]
        path = resolve_path(src, self.project_root)
        try:
            df = pd.read_csv(path)
        except pd.errors.ParserError as e:
            print(f"[datamodule] {path.name}: parser error ({e}); retrying with on_bad_lines='skip'",
                  flush=True)
            df = pd.read_csv(path, on_bad_lines="skip")
        # strip duplicate-suffixed columns from prior CSV patches (e.g. hbond_density.1)
        df = df.loc[:, ~df.columns.str.match(r".*\.\d+$")]
        return df

    def load_features(self, config: dict[str, Any] | None = None) -> pd.DataFrame:
        """Read every relevant source and inner-join on ``file``.

        The join happens in this order so missing rows in one source do not
        silently erase signal from others:

            OpenMM (rest) × COCaDA × ESM  (×  OpenMM (unrest) if mode needs it)

        Returns a single DataFrame keyed on ``file`` with:
          * label         (0 or 1)
          * status        (OK/ERROR — upstream OpenMM status)
          * OPENMM_COLS_27
          * `{c}__unrest` for c in OPENMM_COLS_27 (only for modes needing it)
          * COCADA_COLS_4
          * esm_embed_0 .. esm_embed_{N-1}  (raw ESM embedding, PCA applied later)
        """
        if config is not None:
            self.config = config

        mode = self.config["feature_mode"]
        BOTH_MODES = ("both", "both_delta", "both_raw", "both_all")
        need_rest = True                       # all modes
        need_unrest = mode in ("unrest",) + BOTH_MODES
        need_cocada_rest = mode in ("rest",) + BOTH_MODES
        # both_delta does NOT need unrest-COCaDA (delta only applies to OpenMM).
        need_cocada_unrest = mode in ("unrest", "both", "both_raw", "both_all")
        need_esm = True

        # Load positives + negatives per source, tag with label.
        rest_pos = self._read_csv("openmm_rest").assign(label=1)
        rest_neg = self._read_csv("openmm_rest_neg").assign(label=0)
        rest = pd.concat([rest_pos, rest_neg], ignore_index=True)

        # OpenMM CSVs already carry their own `status` column; preserve it.
        keep_cols = ["file", "label", "status"] + OPENMM_COLS_27
        missing = [c for c in keep_cols if c not in rest.columns]
        if missing:
            raise KeyError(f"OpenMM-rest CSV missing cols: {missing}")
        rest = rest[keep_cols].copy()

        merged = rest

        if need_unrest:
            unr_pos = self._read_csv("openmm_unrest").assign(label=1)
            unr_neg = self._read_csv("openmm_unrest_neg").assign(label=0)
            unr = pd.concat([unr_pos, unr_neg], ignore_index=True)
            unr_cols = ["file", "status"] + OPENMM_COLS_27
            unr = unr[unr_cols].copy()
            # rename feature cols with __unrest suffix; keep status under a dedicated name
            rename_map = {c: unrest_suffix(c) for c in OPENMM_COLS_27}
            rename_map["status"] = "status_unrest"
            unr = unr.rename(columns=rename_map)
            merged = merged.merge(unr, on="file", how="inner")

        if need_cocada_rest:
            coc_pos = self._read_csv("cocada")
            coc_neg = self._read_csv("cocada_neg")
            coc = pd.concat([coc_pos, coc_neg], ignore_index=True)
            coc = coc[["file"] + COCADA_COLS_4].copy()
            merged = merged.merge(coc, on="file", how="inner")

        if need_cocada_unrest:
            coc_pos = self._read_csv("cocada_unrest")
            coc_neg = self._read_csv("cocada_unrest_neg")
            coc = pd.concat([coc_pos, coc_neg], ignore_index=True)
            coc = coc[["file"] + COCADA_COLS_4].copy()
            coc = coc.rename(columns={c: unrest_suffix(c) for c in COCADA_COLS_4})
            merged = merged.merge(coc, on="file", how="inner")

        if need_esm:
            esm_pos = self._read_csv("esm")
            esm_neg = self._read_csv("esm_neg")
            esm = pd.concat([esm_pos, esm_neg], ignore_index=True)
            esm_feat_cols = [c for c in esm.columns if c != "file"]
            esm = esm[["file"] + esm_feat_cols].copy()
            merged = merged.merge(esm, on="file", how="inner")
            self._esm_feat_cols = esm_feat_cols
        else:
            self._esm_feat_cols = []

        # drop duplicate files defensively (prefer first occurrence)
        merged = merged.drop_duplicates(subset=["file"], keep="first").reset_index(drop=True)
        return merged

    # ── Hold-out / error filtering ───────────────────────────────────────
    def apply_holdout(
        self,
        df: pd.DataFrame,
        held_out_orphans_tsv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Remove rows listed in ``held_out_orphans.tsv`` from the training df.

        The held-out orphans are never used for training — they exist
        solely to support the post-train OOD benchmark (``ood_eval``).
        """
        if held_out_orphans_tsv is None:
            held_out_orphans_tsv = self.config["hold_out"]["held_out_orphans_tsv"]
        path = resolve_path(held_out_orphans_tsv, self.project_root)
        if not path.exists():
            # no-op if file is absent (still-scaffolding scenario)
            return df
        tsv = pd.read_csv(path, sep="\t")
        if "file" not in tsv.columns:
            raise KeyError(f"{path} lacks 'file' column")
        held = set(tsv["file"].astype(str))
        before = len(df)
        out = df[~df["file"].isin(held)].reset_index(drop=True)
        print(f"[datamodule] apply_holdout: removed {before - len(out)} held-out orphans "
              f"(expected up to {len(held)})", flush=True)
        return out

    def drop_error_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where upstream OpenMM extraction failed (status != OK)."""
        before = len(df)
        mask = df["status"].astype(str).str.upper().str.startswith("OK")
        if "status_unrest" in df.columns:
            mask = mask & df["status_unrest"].astype(str).str.upper().str.startswith("OK")
        out = df[mask].reset_index(drop=True)
        print(f"[datamodule] drop_error_rows: removed {before - len(out)} non-OK rows", flush=True)
        return out

    # ── Labels ───────────────────────────────────────────────────────────
    @staticmethod
    def build_labels(df: pd.DataFrame) -> np.ndarray:
        """Return the binary label array (positives=1, negatives=0).

        Labels are carried on the merged df from the positive/negative source
        tagging in ``load_features``.  This helper also validates that the
        assignment agrees with the filename convention: a file whose name
        starts with ``neg_`` must have label 0, anything else must be 1.
        """
        if "label" not in df.columns:
            raise KeyError("build_labels: df must contain 'label'")
        labels = df["label"].astype(int).values
        # sanity check
        is_neg_by_name = df["file"].astype(str).str.startswith("neg_").values
        mismatch = (labels == 1) & is_neg_by_name
        if mismatch.any():
            bad = df.loc[mismatch, "file"].tolist()[:5]
            raise ValueError(
                f"build_labels: label=1 rows whose filename starts with 'neg_' "
                f"(first 5): {bad}"
            )
        mismatch2 = (labels == 0) & ~is_neg_by_name
        if mismatch2.any():
            bad = df.loc[mismatch2, "file"].tolist()[:5]
            raise ValueError(
                f"build_labels: label=0 rows whose filename is NOT a negative "
                f"(first 5): {bad}"
            )
        return labels

    # ── ESM PCA ──────────────────────────────────────────────────────────
    def esm_pca(
        self,
        X: np.ndarray,
        n_dims: int | None = None,
        fit: bool = True,
    ) -> np.ndarray:
        """Fit (or reuse) a PCA on the ESM embedding block and project.

        ``fit=True`` fits a fresh PCA on ``X`` and caches it on the instance.
        ``fit=False`` applies the cached projection (used for LOO folds where
        the PCA is fit once on the full cohort so every fold sees a
        consistent embedding space).

        We do NOT refit per-fold by default: the ESM embedding is a
        sample-level sequence representation, so leakage risk is minimal
        compared to the feature-to-label signal, and refitting per fold would
        destabilise LOO significantly for N=3000.
        """
        if n_dims is None:
            n_dims = int(self.config.get("esm_pca_dims", 64))
        X = np.asarray(X, dtype=np.float32)
        if fit or self._pca is None:
            # standardise first (PCA is variance-sensitive)
            mean = X.mean(axis=0)
            std = X.std(axis=0) + 1e-8
            Xs = (X - mean) / std
            pca = PCA(n_components=n_dims, random_state=42)
            Z = pca.fit_transform(Xs).astype(np.float32)
            self._pca = pca
            self._esm_scaler_mean = mean
            self._esm_scaler_std = std
            return Z
        # reuse cached scaler + PCA
        Xs = (X - self._esm_scaler_mean) / self._esm_scaler_std
        return self._pca.transform(Xs).astype(np.float32)

    def attach_esm_pca(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Replace raw ESM embedding columns with ``esm_pca_*`` columns.

        Called once per training run (fit=True) or once per LOO campaign
        (fit=True on the full cohort; subsequent folds reuse via fit=False).
        """
        if not self._esm_feat_cols:
            return df
        X = df[self._esm_feat_cols].values.astype(np.float32)
        n_dims = int(self.config.get("esm_pca_dims", 64))
        Z = self.esm_pca(X, n_dims=n_dims, fit=fit)
        pca_cols = esm_pca_cols(n_dims)
        out = df.drop(columns=self._esm_feat_cols).copy()
        for i, c in enumerate(pca_cols):
            out[c] = Z[:, i]
        return out

    # ── Convenience: full pipeline ───────────────────────────────────────
    def prepare(self) -> pd.DataFrame:
        """Run the full load → hold-out → error-drop → PCA pipeline."""
        df = self.load_features()
        df = self.apply_holdout(df)
        if self.config.get("hold_out", {}).get("drop_error_rows", True):
            df = self.drop_error_rows(df)
        df = self.attach_esm_pca(df, fit=True)
        return df
