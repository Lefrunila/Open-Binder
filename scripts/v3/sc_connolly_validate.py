#!/usr/bin/env python3
"""sc_connolly_validate.py — Validate SES-based SC against Rosetta MSMS values.

Reads the first 50 structures from features_positives_openmm_v2.csv,
computes SC via the grid-based SES method, and reports:
  - Pearson r, Spearman rho, MAE vs Rosetta sc column
  - Scatter plot saved to Open-Binder/docs/figures/sc_validation.pdf
  - Per-variant comparison (interface_cutoff, grid_spacing, agg function)

Usage
-----
    python sc_connolly_validate.py
    python sc_connolly_validate.py --n-structures 50 --workers 4
    python sc_connolly_validate.py --grid-spacing 0.3 --n-structures 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Add the script dir to path so we can import sc_connolly
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

import sc_connolly  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent  # Open-Binder/
CSV_PATH = BASE_DIR / "data" / "features_positives_openmm_v2.csv"
# PDB_DIR: set via --pdb-dir CLI argument or the OPENBINDER_PDB_DIR env var.
# Defaults to data/structures/positives_cleaned/ (unpack positives_cleaned.tar.gz first).
import os as _os
PDB_DIR = Path(_os.environ.get(
    "OPENBINDER_PDB_DIR",
    str(BASE_DIR / "data" / "structures" / "positives_cleaned"),
))
FIG_DIR = BASE_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_antigen_chain(pdb_path: str) -> str:
    """Return the non-H chain ID from a PDB (antigen chain)."""
    from Bio import PDB
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("x", pdb_path)
        model = next(structure.get_models())
        for chain in model:
            if chain.id != "H":
                return chain.id
    except Exception:
        pass
    return "A"  # fallback


def _run_one(args_tuple):
    """Worker: compute SC for one structure. Returns (file, sc_ses, elapsed)."""
    f, pdb_path, kwargs = args_tuple
    t0 = time.time()
    try:
        ag_chain = _get_antigen_chain(pdb_path)
        sc = sc_connolly.compute_sc(
            pdb_path=pdb_path,
            chain_vhh="H",
            chain_ag=ag_chain,
            **kwargs,
        )
    except Exception as exc:
        print(f"  ERROR {f}: {exc}", file=sys.stderr)
        sc = float("nan")
    elapsed = time.time() - t0
    return f, sc, elapsed


def run_validation(
    n_structures: int = 50,
    grid_spacing: float = 0.5,
    interface_cutoff: float = 3.5,
    interface_cutoff_prefilter: float = 2.5,
    workers: int = 4,
    label: str = "",
) -> dict:
    """Run validation and return result dict."""
    df_all = pd.read_csv(CSV_PATH)
    # Filter to valid sc values
    df_valid = df_all[df_all["sc"].notna() & (df_all["status"] == "OK")].copy()
    df_sample = df_valid.head(n_structures).copy()
    print(f"\n{'='*60}")
    print(f"Variant: {label or 'default'}")
    print(f"  grid_spacing={grid_spacing}  interface_cutoff={interface_cutoff}  n={len(df_sample)}")

    kwargs = dict(
        grid_spacing=grid_spacing,
        interface_cutoff=interface_cutoff,
        interface_cutoff_prefilter=interface_cutoff_prefilter,
    )

    tasks = []
    for f in df_sample["file"]:
        pdb_path = str(PDB_DIR / f)
        if not os.path.exists(pdb_path):
            print(f"  MISSING: {pdb_path}", file=sys.stderr)
            continue
        tasks.append((f, pdb_path, kwargs))

    sc_ses_map = {}
    timings = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            f, sc, elapsed = fut.result()
            sc_ses_map[f] = sc
            timings.append(elapsed)
            done += 1
            print(f"  [{done}/{len(tasks)}] {f}: SC={sc:.4f}  ({elapsed:.1f}s)")

    # Merge with Rosetta values
    df_sample["sc_ses"] = df_sample["file"].map(sc_ses_map)
    df_cmp = df_sample.dropna(subset=["sc", "sc_ses"])

    if len(df_cmp) < 5:
        print(f"  Too few valid pairs ({len(df_cmp)}), cannot compute correlation")
        return {"label": label, "n": len(df_cmp), "pearson_r": float("nan"),
                "spearman_rho": float("nan"), "mae": float("nan"),
                "mean_time": float("nan"), "df": df_cmp}

    rosetta = df_cmp["sc"].values
    ses = df_cmp["sc_ses"].values

    r, pval_r = pearsonr(rosetta, ses)
    rho, pval_rho = spearmanr(rosetta, ses)
    mae = float(np.mean(np.abs(rosetta - ses)))
    mean_t = float(np.mean(timings)) if timings else float("nan")

    print(f"\n  *** Results ({label}) ***")
    print(f"  N pairs       = {len(df_cmp)}")
    print(f"  Pearson r     = {r:.4f}  (p={pval_r:.2e})")
    print(f"  Spearman rho  = {rho:.4f}  (p={pval_rho:.2e})")
    print(f"  MAE           = {mae:.4f}")
    print(f"  Mean time/str = {mean_t:.1f}s")

    return {
        "label": label,
        "n": len(df_cmp),
        "pearson_r": r,
        "spearman_rho": rho,
        "mae": mae,
        "mean_time": mean_t,
        "df": df_cmp,
    }


def make_scatter(results_list: list, out_path: Path) -> None:
    """Save a multi-panel scatter plot comparing variants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr

    n_panels = len(results_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), squeeze=False)
    axes = axes[0]

    for ax, res in zip(axes, results_list):
        if res["df"] is None or len(res["df"]) < 2:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center")
            continue
        df = res["df"]
        x = df["sc"].values
        y = df["sc_ses"].values
        ax.scatter(x, y, alpha=0.6, s=30, edgecolors="none", color="steelblue")
        # 1:1 line
        lo, hi = min(x.min(), y.min()) - 0.05, max(x.max(), y.max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Rosetta SC (MSMS)")
        ax.set_ylabel("OpenBinder SC (SES)")
        r = res["pearson_r"]
        rho = res["spearman_rho"]
        n = res["n"]
        ax.set_title(f"{res['label']}\nr={r:.3f}  ρ={rho:.3f}  N={n}", fontsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\nScatter plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate SES-based SC vs Rosetta")
    parser.add_argument("--n-structures", type=int, default=50,
                        help="Number of structures to validate (default 50)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid-spacing", type=float, default=0.5,
                        dest="grid_spacing")
    parser.add_argument("--interface-cutoff", type=float, default=3.5,
                        dest="interface_cutoff")
    parser.add_argument("--skip-variants", action="store_true",
                        help="Skip variant comparison, run only default config")
    parser.add_argument("--pdb-dir", type=Path, default=None,
                        help="Directory of cleaned PDB structures to validate against. "
                             "Overrides the OPENBINDER_PDB_DIR env var. "
                             "Unpack data/structures/positives_cleaned.tar.gz first.")
    args = parser.parse_args()

    # Apply --pdb-dir override
    if args.pdb_dir is not None:
        global PDB_DIR
        PDB_DIR = args.pdb_dir.resolve()

    all_results = []

    # ---- Optimal config (prefilter=2.5A, ic=3.5A, median_pos) ---------------
    res0 = run_validation(
        n_structures=args.n_structures,
        grid_spacing=args.grid_spacing,
        interface_cutoff=args.interface_cutoff,
        interface_cutoff_prefilter=2.5,
        workers=args.workers,
        label=f"gs={args.grid_spacing} pf=2.5 ic={args.interface_cutoff}",
    )
    all_results.append(res0)

    if not args.skip_variants and args.n_structures >= 20:
        # ---- Variant: broader surface-surface cutoff -------------------------
        res1 = run_validation(
            n_structures=args.n_structures,
            grid_spacing=args.grid_spacing,
            interface_cutoff=3.0,
            interface_cutoff_prefilter=2.5,
            workers=args.workers,
            label=f"gs={args.grid_spacing} pf=2.5 ic=3.0",
        )
        all_results.append(res1)

        # ---- Variant: looser atom prefilter ----------------------------------
        res2 = run_validation(
            n_structures=args.n_structures,
            grid_spacing=args.grid_spacing,
            interface_cutoff=3.5,
            interface_cutoff_prefilter=5.0,
            workers=args.workers,
            label=f"gs={args.grid_spacing} pf=5.0 ic=3.5",
        )
        all_results.append(res2)

    # ---- Best result ---------------------------------------------------------
    best = max(all_results, key=lambda r: r["pearson_r"] if not np.isnan(r["pearson_r"]) else -1)
    print(f"\n{'='*60}")
    print(f"BEST variant: {best['label']}")
    print(f"  Pearson r    = {best['pearson_r']:.4f}")
    print(f"  Spearman rho = {best['spearman_rho']:.4f}")
    print(f"  MAE          = {best['mae']:.4f}")
    print(f"  Mean time    = {best['mean_time']:.1f}s/structure")

    if best["pearson_r"] >= 0.80:
        print("\n  SUCCESS: r >= 0.80 — implementation ready for full re-extraction.")
    else:
        print(f"\n  WARNING: r = {best['pearson_r']:.4f} < 0.80 — further tuning needed.")

    # ---- Scatter plot --------------------------------------------------------
    fig_path = FIG_DIR / "sc_validation.pdf"
    make_scatter(all_results, fig_path)

    return best["pearson_r"]


if __name__ == "__main__":
    main()
