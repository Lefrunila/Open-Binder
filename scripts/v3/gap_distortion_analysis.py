#!/usr/bin/env python3
"""gap_distortion_analysis.py — Empirical test of gap-edge anchor restraint load-bearingness.

Compares CA coordinates of structures relaxed under the two relaxation states
(restrained vs unrestrained), measuring RMSD over (a) the whole structure,
(b) antigen-only CAs, and (c) a window of residues flanking each rebuilt gap
edge. Used to validate the §2.3 claim that the strong gap-flanking anchor
restraints (k=5000 kJ/mol/nm^2) applied in the restrained protocol are not
load-bearing — i.e., the AMBER14/OBC2 force field's intrinsic regularization
keeps the gap edges in place even without explicit anchors.

Sampling: positives are ranked by their largest gap span and a representative
sample of 21 structures is drawn (2 smallest, 2 small-mid, 2 mid, 2 large,
2 largest, plus every 25th in the ranking); only structures with a
``<stem>_gap_split.json`` sidecar are eligible.

Reported in §2.3 of the manuscript: gap-edge Cα RMSD median 0.64 Å vs
whole-structure median 0.63 Å (N = 21).

Usage
-----
    python scripts/v3/gap_distortion_analysis.py \
        --positives-dir /path/to/positives \
        --rest-dir      /path/to/positives_relaxed \
        --unrest-dir    /path/to/positives_unrest

The three directories must contain matching ``<stem>.pdb`` files; the
``--positives-dir`` must additionally hold the ``<stem>_gap_split.json``
sidecars produced by the upstream gap-split step.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore")

WINDOW = 5  # residues each side of gap edge


def load_ca(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Return {(chain_id, resseq) -> CA coord (3,)} for the first model."""
    s = PDBParser(QUIET=True).get_structure(path.stem, str(path))
    out: dict[tuple[str, int], np.ndarray] = {}
    for model in s:
        for chain in model:
            cid = chain.id
            for res in chain:
                if res.id[0] != " ":
                    continue
                if "CA" in res:
                    out[(cid, res.id[1])] = res["CA"].coord.astype(float)
        break
    return out


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Optimal rigid alignment of P -> Q. Returns (R, t, whole_rmsd, P_aligned)."""
    Pc = P.mean(0)
    Qc = Q.mean(0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    P_aligned = (P0 @ R.T) + Qc
    rmsd = float(np.sqrt(((P_aligned - Q) ** 2).sum(axis=1).mean()))
    return R, Qc - Pc @ R.T, rmsd, P_aligned


def rmsd_subset(R: np.ndarray, P_full: np.ndarray, Q_full: np.ndarray, mask: np.ndarray) -> float:
    Pc = P_full.mean(0)
    Qc = Q_full.mean(0)
    P_al = (P_full - Pc) @ R.T + Qc
    diff = P_al[mask] - Q_full[mask]
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def collect_gap_edge_keys(splits, ca_keys):
    """Take WINDOW residues each side of every left/right gap-anchor."""
    keys: set[tuple[str, int]] = set()
    by_chain: dict[str, list[int]] = {}
    for c, r in ca_keys:
        by_chain.setdefault(c, []).append(r)
    for c in by_chain:
        by_chain[c].sort()
    for sp in splits:
        for side in ("left", "right"):
            cid = sp[f"{side}_chain"]
            anc = sp[f"{side}_anchor_resnum"]
            if cid not in by_chain:
                continue
            sorted_r = by_chain[cid]
            if anc not in sorted_r:
                anc = min(sorted_r, key=lambda r: abs(r - anc))
            idx = sorted_r.index(anc)
            lo = max(0, idx - WINDOW)
            hi = min(len(sorted_r), idx + WINDOW + 1)
            for r in sorted_r[lo:hi]:
                keys.add((cid, r))
    return keys


def antigen_keys(ca_keys):
    """Antigen = every chain that is not 'H' (chain H is always the VHH)."""
    return {k for k in ca_keys if k[0] != "H"}


def analyse(stem: str, positives_dir: Path, rest_dir: Path, unrest_dir: Path):
    rest_pdb = rest_dir / f"{stem}.pdb"
    unrest_pdb = unrest_dir / f"{stem}.pdb"
    sidecar = positives_dir / f"{stem}_gap_split.json"
    if not (rest_pdb.exists() and unrest_pdb.exists() and sidecar.exists()):
        return None
    splits = json.loads(sidecar.read_text())["splits"]
    rest = load_ca(rest_pdb)
    unrest = load_ca(unrest_pdb)
    common = sorted(set(rest.keys()) & set(unrest.keys()))
    if len(common) < 30:
        return None
    P = np.array([unrest[k] for k in common])
    Q = np.array([rest[k] for k in common])
    R, _, whole_rmsd, _ = kabsch(P, Q)
    ant_keys = antigen_keys(common)
    gap_keys = collect_gap_edge_keys(splits, common)
    if not gap_keys:
        return None
    ant_mask = np.array([k in ant_keys for k in common])
    gap_mask = np.array([k in gap_keys for k in common])
    return {
        "stem": stem,
        "n_gaps": len(splits),
        "n_ca": len(common),
        "n_gap_edge_ca": int(gap_mask.sum()),
        "whole_rmsd": whole_rmsd,
        "antigen_rmsd": rmsd_subset(R, P, Q, ant_mask) if ant_mask.any() else float("nan"),
        "gap_edge_rmsd": rmsd_subset(R, P, Q, gap_mask) if gap_mask.any() else float("nan"),
        "max_gap_size": max(sp["gap_end"] - sp["gap_start"] + 1 for sp in splits),
    }


def rank_by_gap_size(positives_dir: Path):
    rows = []
    for sc in positives_dir.glob("*_gap_split.json"):
        try:
            d = json.loads(sc.read_text())
        except Exception:
            continue
        gaps = [sp["gap_end"] - sp["gap_start"] + 1 for sp in d.get("splits", [])]
        if not gaps:
            continue
        rows.append((sc.stem.replace("_gap_split", ""), max(gaps), len(gaps)))
    rows.sort(key=lambda r: r[1])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-dir", type=Path, required=True,
                        help="Directory containing <stem>.pdb and <stem>_gap_split.json sidecars.")
    parser.add_argument("--rest-dir", type=Path, required=True,
                        help="Directory of restrained-relaxation outputs.")
    parser.add_argument("--unrest-dir", type=Path, required=True,
                        help="Directory of unrestrained-relaxation outputs.")
    args = parser.parse_args()

    ranked = rank_by_gap_size(args.positives_dir)
    n = len(ranked)
    if n < 10:
        sample_stems = [r[0] for r in ranked]
    else:
        idxs = [0, 1, n // 4, n // 4 + 1, n // 2, n // 2 + 1,
                3 * n // 4, 3 * n // 4 + 1, n - 2, n - 1]
        sample_stems = [ranked[i][0] for i in idxs]
    extended = sample_stems + [ranked[i][0] for i in range(2, n, 25)
                               if ranked[i][0] not in sample_stems]
    sample_stems = extended

    print(f"Total positives with gap_split: {n}")
    print(f"Selected sample of {len(sample_stems)}\n")
    print(f"{'stem':30s} {'max_gap':>7s} {'n_gaps':>6s} {'n_ca':>5s} "
          f"{'gap_edge_n':>10s} {'whole':>7s} {'antigen':>7s} {'gap_edge':>8s}")
    results = []
    for stem in sample_stems:
        r = analyse(stem, args.positives_dir, args.rest_dir, args.unrest_dir)
        if r is None:
            print(f"{stem:30s} -- skipped (missing input)")
            continue
        print(f"{r['stem']:30s} {r['max_gap_size']:7d} {r['n_gaps']:6d} "
              f"{r['n_ca']:5d} {r['n_gap_edge_ca']:10d}  {r['whole_rmsd']:6.3f}  "
              f"{r['antigen_rmsd']:6.3f}  {r['gap_edge_rmsd']:7.3f}")
        results.append(r)

    print()
    if results:
        gers = np.array([r["gap_edge_rmsd"] for r in results])
        wrs = np.array([r["whole_rmsd"] for r in results])
        ars = np.array([r["antigen_rmsd"] for r in results])
        print(f"Gap-edge RMSD: median={np.median(gers):.3f}  "
              f"IQR=({np.percentile(gers, 25):.3f}, {np.percentile(gers, 75):.3f})  "
              f"max={gers.max():.3f}")
        print(f"Whole  RMSD : median={np.median(wrs):.3f}  "
              f"IQR=({np.percentile(wrs, 25):.3f}, {np.percentile(wrs, 75):.3f})  "
              f"max={wrs.max():.3f}")
        print(f"Antigen RMSD: median={np.median(ars):.3f}  "
              f"IQR=({np.percentile(ars, 25):.3f}, {np.percentile(ars, 75):.3f})  "
              f"max={ars.max():.3f}")


if __name__ == "__main__":
    main()
