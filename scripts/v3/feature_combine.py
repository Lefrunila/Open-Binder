#!/usr/bin/env python3
"""feature_combine.py

Stateless helpers for assembling feature matrices from the canonical column
families.  Used by ``datamodule``, ``rf_train``, ``mlp_train``, and the LOO
harness so that every entry point sees an identical column ordering.

The three modes (strictly symmetric apples-to-apples):

    rest   → 27 OpenMM_rest   + 4 COCaDA_rest   + ESM-PCA              = 95 feats
    unrest → 27 OpenMM_unrest + 4 COCaDA_unrest + ESM-PCA              = 95 feats
    both   → 27 OpenMM_rest + 27 OpenMM_unrest + 4 COCaDA_rest
             + 4 COCaDA_unrest + ESM-PCA                               = 126 feats (RF)
             for MLP "both" we instead expose two branches via ``two_branch_input``

ESM is sequence-only, so it is identical across relax modes and appears once.
OpenMM and COCaDA are structure-derived, so each mode uses the variants matching
its relaxation. No pre-computed deltas: RF and MLP can recover any difference
from the raw rest/unrest pair.

No side effects, no config parsing.  Pure numpy / pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Canonical column families ────────────────────────────────────────────────
OPENMM_COLS_27 = [
    "complex_normalized", "dG_cross", "dG_cross/dSASAx100", "dSASA_hphobic",
    "dSASA_int", "dSASA_polar", "delta_unsatHbonds", "dslf_fa13", "fa_atr",
    "hbond_E_fraction", "hbond_bb_sc", "hbond_lr_bb", "hbond_sc", "hbond_sr_bb",
    "hbonds_int", "nres_int", "omega", "pro_close",
    "rama_prepro", "ref", "yhh_planarity", "sc",
    "nres_int_vhh", "nres_int_ag", "dSASA_vhh", "dSASA_ag", "hbond_density",
]

COCADA_COLS_4 = [
    "n_salt_bridges_int", "n_aromatic_stacking_int",
    "n_hydrophobic_int", "n_repulsive_int",
]


def rest_suffix(col: str) -> str:
    """Return the restrained-variant column name (identity — rest is the base)."""
    return col


def unrest_suffix(col: str) -> str:
    """Return the unrestrained-variant column name (the ``_unrest`` variant)."""
    return f"{col}__unrest"


def delta_suffix(col: str) -> str:
    """Return the delta column name used by RF ``both`` mode."""
    return f"{col}__delta"


def esm_pca_cols(n_dims: int) -> list[str]:
    """Canonical ESM-PCA column names (``esm_pca_0`` … ``esm_pca_{n-1}``)."""
    return [f"esm_pca_{i}" for i in range(n_dims)]


# ── Column ordering by mode ──────────────────────────────────────────────────
def feature_cols_rest(esm_dims: int) -> list[str]:
    """Column ordering for mode=rest (RF or MLP)."""
    return list(OPENMM_COLS_27) + list(COCADA_COLS_4) + esm_pca_cols(esm_dims)


def feature_cols_unrest(esm_dims: int) -> list[str]:
    """Column ordering for mode=unrest (27 OpenMM_unrest + 4 COCaDA_unrest + ESM)."""
    unrest_openmm = [unrest_suffix(c) for c in OPENMM_COLS_27]
    unrest_cocada = [unrest_suffix(c) for c in COCADA_COLS_4]
    return unrest_openmm + unrest_cocada + esm_pca_cols(esm_dims)


def feature_cols_both_rf(esm_dims: int, variant: str = "all") -> list[str]:
    """Column ordering for RF mode=both*, parameterized by ``variant``.

    Variants (for paper comparison):
      * ``"delta"`` → 27 OpenMM_rest + 27 OpenMM_delta + 4 COCaDA_rest + ESM  (122)
      * ``"raw"``   → 27 OpenMM_rest + 27 OpenMM_unrest + 4 COCaDA_rest
                      + 4 COCaDA_unrest + ESM                               (126)
      * ``"all"``   → 27 OpenMM_rest + 27 OpenMM_unrest + 27 OpenMM_delta
                      + 4 COCaDA_rest + 4 COCaDA_unrest + ESM               (153)
    """
    rest_openmm = list(OPENMM_COLS_27)
    unrest_openmm = [unrest_suffix(c) for c in OPENMM_COLS_27]
    delta_openmm = [delta_suffix(c) for c in OPENMM_COLS_27]
    rest_cocada = list(COCADA_COLS_4)
    unrest_cocada = [unrest_suffix(c) for c in COCADA_COLS_4]
    esm = esm_pca_cols(esm_dims)
    if variant == "delta":
        return rest_openmm + delta_openmm + rest_cocada + esm
    if variant == "raw":
        return rest_openmm + unrest_openmm + rest_cocada + unrest_cocada + esm
    if variant == "all":
        return (rest_openmm + unrest_openmm + delta_openmm
                + rest_cocada + unrest_cocada + esm)
    raise ValueError(f"feature_cols_both_rf: unknown variant {variant!r}")


# ── Delta / two-branch helpers ───────────────────────────────────────────────
def compute_delta(rest_cols: np.ndarray, unrest_cols: np.ndarray) -> np.ndarray:
    """Return ``rest`` concatenated with ``(unrest - rest)`` along axis=1.

    Parameters
    ----------
    rest_cols : (n, k) ndarray
    unrest_cols : (n, k) ndarray

    Returns
    -------
    (n, 2k) ndarray
    """
    rest = np.asarray(rest_cols, dtype=np.float32)
    unrest = np.asarray(unrest_cols, dtype=np.float32)
    if rest.shape != unrest.shape:
        raise ValueError(
            f"compute_delta: shape mismatch rest={rest.shape} unrest={unrest.shape}"
        )
    return np.concatenate([rest, unrest - rest], axis=1)


def two_branch_input(
    rest_cols: np.ndarray,
    unrest_cols: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return two-branch input dict for the MLP ``both`` mode."""
    return {
        "rest": np.asarray(rest_cols, dtype=np.float32),
        "unrest": np.asarray(unrest_cols, dtype=np.float32),
    }


# ── DataFrame assembly (used by datamodule after merges) ─────────────────────
def assemble_matrix(
    df: pd.DataFrame,
    mode: str,
    esm_dims: int,
) -> tuple[np.ndarray, list[str]]:
    """Pull the feature matrix out of a merged DataFrame.

    ``df`` is expected to already contain (at minimum):

      * For mode=rest:   OPENMM_COLS_27 + COCADA_COLS_4 + esm_pca_{i}
      * For mode=unrest: ``{c}__unrest`` (OPENMM_COLS_27 + COCADA_COLS_4) + esm_pca_{i}
      * For mode=both:   OPENMM_COLS_27 + ``{c}__unrest`` (OPENMM_COLS_27)
                         + COCADA_COLS_4 + ``{c}__unrest`` (COCADA_COLS_4) + esm_pca_{i}

    The function does NOT drop NaNs — caller handles that.
    """
    if mode == "rest":
        cols = feature_cols_rest(esm_dims)
    elif mode == "unrest":
        cols = feature_cols_unrest(esm_dims)
    elif mode in ("both", "both_delta", "both_raw", "both_all"):
        variant = "all" if mode == "both" else mode.split("_", 1)[1]
        if variant in ("delta", "all"):
            df = df.copy()
            for c in OPENMM_COLS_27:
                df[delta_suffix(c)] = df[unrest_suffix(c)].values - df[c].values
        cols = feature_cols_both_rf(esm_dims, variant=variant)
    else:
        raise ValueError(f"assemble_matrix: unknown mode {mode!r}")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"assemble_matrix: missing {len(missing)} columns for mode={mode!r}: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
        )
    return df[cols].values.astype(np.float32), cols
