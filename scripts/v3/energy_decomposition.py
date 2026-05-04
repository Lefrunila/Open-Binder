#!/usr/bin/env python3
"""energy_decomposition.py — Per-term AMBER14/OBC2 energy decomposition for rest vs unrest relaxed PDBs.

Reproduces the empirical investigation cited in §2.4 P38 of the OpenBinder
manuscript: when CA position restraints are released during the unrestrained
relaxation, the OBC2 implicit-solvent gradient pulls polar atoms outward, so
the unrestrained state ends up at a *different* (higher total energy) basin
than the restrained one — vacuum energy (Coulomb + LJ) goes up, GBSA goes
down. The discriminative signal in delta(fa_atr) reflects the differential
magnitude of this restraint-release drift between native binder and
VHH-swap decoy interfaces.

Procedure:
  1. Sample N_binders + N_decoys representative structures per class, spanning
     the 5/30/50/70/95th percentiles of delta(fa_atr) = fa_atr_unrest - fa_atr_rest.
  2. For each sampled structure and both relaxation states, single-point energy
     using the same force field as the relaxation step (amber14-all + obc2)
     with NO minimization and NO restraint forces.
  3. Decompose the total potential into bond, angle, torsion, NonbondedForce
     (Coulomb + full LJ + 1-4), and GBSA terms via OpenMM force groups.
     Additionally split the LJ van-der-Waals term into its attractive (V<0)
     and repulsive (V>0) components via two CustomNonbondedForce probes,
     matching the fa_atr / fa_rep convention used in scripts/v3/feature_combine.py.

Reported in §2.4 P38: a 5+5 representative sample shows fa_atr and GBSA
anti-correlated r = -0.73, confirming a packing-vs-solvation trade-off; the
unrestrained state has higher total energy than the restrained state in 9/10
structures.

Usage
-----
    python scripts/v3/energy_decomposition.py \\
        --pos-feat-rest    data/features_positives_openmm_v2.csv \\
        --pos-feat-unrest  data/features_positives_openmm_unrest.csv \\
        --neg-feat-rest    data/features_negatives_openmm_v2.csv \\
        --neg-feat-unrest  data/features_negatives_openmm_unrest.csv \\
        --pos-pdb-rest     /path/to/positives_relaxed \\
        --pos-pdb-unrest   /path/to/positives_unrest \\
        --neg-pdb-rest     /path/to/negatives_relaxed \\
        --neg-pdb-unrest   /path/to/negatives_unrest \\
        --output-csv       docs/energy_decomposition.csv

The PDB directories must contain matching `<file>.pdb` filenames for both
relaxation states. The feature CSVs supply the `fa_atr` column and the `file`
key used for sampling.
"""
from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import openmm as mm
import openmm.app as app
import openmm.unit as unit
from pdbfixer import PDBFixer

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "CYX", "ACE", "NME",
}

PERCENTILES = [5, 30, 50, 70, 95]


def sample_by_percentile(df: pd.DataFrame, n: int, delta_col: str = "delta_fa_atr") -> pd.DataFrame:
    """Pick `n` rows from `df` spanning the 5/30/50/70/95th percentiles of delta_fa_atr."""
    df = df.sort_values(delta_col).reset_index(drop=True)
    if len(df) <= n:
        return df
    qs = np.linspace(0, 100, n + 2)[1:-1] if n != 5 else PERCENTILES
    idxs = [int(round((q / 100) * (len(df) - 1))) for q in qs]
    return df.iloc[idxs].reset_index(drop=True)


def load_pdb_minimal(pdb_path: Path):
    """Load a relaxed PDB through PDBFixer (hydrogens already present)."""
    raw = pdb_path.read_text().splitlines()
    keep = []
    for line in raw:
        rec = line[:6].strip()
        if rec in ("ATOM", "TER"):
            if rec == "ATOM" and len(line) > 20:
                resname = line[17:20].strip()
                if resname == "UNK":
                    continue
            keep.append(line)
    fixer = PDBFixer(pdbfile=StringIO("\n".join(keep)))
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    return fixer.topology, fixer.positions


def decompose_one(pdb_path: Path) -> dict:
    """Return per-term potential energies in kcal/mol for a single PDB."""
    topology, positions = load_pdb_minimal(pdb_path)

    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=app.HBonds,
        rigidWater=True,
    )

    force_groups: dict[str, int] = {}
    next_group = 1
    for f in system.getForces():
        cname = f.__class__.__name__
        if cname not in force_groups:
            force_groups[cname] = next_group
            next_group += 1
        f.setForceGroup(force_groups[cname])

    nb = next(f for f in system.getForces()
              if f.__class__.__name__ == "NonbondedForce")

    def _make_lj_probe(formula: str, group_id: int) -> mm.CustomNonbondedForce:
        cf = mm.CustomNonbondedForce(
            f"{formula};"
            "epsilon12 = sqrt(epsilon1*epsilon2);"
            "sigma12 = 0.5*(sigma1+sigma2)"
        )
        cf.addPerParticleParameter("epsilon")
        cf.addPerParticleParameter("sigma")
        for i in range(nb.getNumParticles()):
            _, s, e = nb.getParticleParameters(i)
            cf.addParticle([
                e.value_in_unit(unit.kilojoule_per_mole),
                s.value_in_unit(unit.nanometer),
            ])
        cf.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
        for i in range(nb.getNumExceptions()):
            p1, p2, *_ = nb.getExceptionParameters(i)
            cf.addExclusion(p1, p2)
        cf.setForceGroup(group_id)
        return cf

    atr_group = next_group
    rep_group = next_group + 1
    cf_atr = _make_lj_probe(
        "min(0, epsilon12*((sigma12/r)^12 - 2*(sigma12/r)^6))", atr_group)
    cf_rep = _make_lj_probe(
        "max(0, epsilon12*((sigma12/r)^12 - 2*(sigma12/r)^6))", rep_group)
    system.addForce(cf_atr)
    system.addForce(cf_rep)

    plat = mm.Platform.getPlatformByName("CPU")
    integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
    sim = app.Simulation(topology, system, integrator, plat)
    sim.context.setPositions(positions)

    def _e(groups):
        st = sim.context.getState(getEnergy=True, groups=groups)
        return st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    bond = force_groups.get("HarmonicBondForce")
    angle = force_groups.get("HarmonicAngleForce")
    tors = [g for cn, g in force_groups.items() if "Torsion" in cn]
    nb_gid = force_groups.get("NonbondedForce")
    gb_gid = (force_groups.get("GBSAOBCForce")
              or force_groups.get("CustomGBForce"))
    base_groups = set(force_groups.values())

    return {
        "file": pdb_path.name,
        "bond_kcal":      _e({bond})  if bond  is not None else 0.0,
        "angle_kcal":     _e({angle}) if angle is not None else 0.0,
        "torsion_kcal":   sum(_e({g}) for g in tors) if tors else 0.0,
        "nonbonded_kcal": _e({nb_gid}) if nb_gid is not None else 0.0,
        "gbsa_kcal":      _e({gb_gid}) if gb_gid is not None else 0.0,
        "fa_atr_kcal":    _e({atr_group}),
        "fa_rep_kcal":    _e({rep_group}),
        "total_kcal":     _e(base_groups),
    }


def build_sample(pos_rest_csv: Path, pos_unrest_csv: Path,
                 neg_rest_csv: Path, neg_unrest_csv: Path,
                 n_binders: int, n_decoys: int) -> pd.DataFrame:
    """Merge rest+unrest features per class and sample by delta(fa_atr) percentile."""
    rows = []
    for label, rest_csv, unrest_csv, n in [
        ("binder", pos_rest_csv, pos_unrest_csv, n_binders),
        ("decoy",  neg_rest_csv, neg_unrest_csv, n_decoys),
    ]:
        rest = pd.read_csv(rest_csv)[["file", "fa_atr"]]
        unrest = pd.read_csv(unrest_csv)[["file", "fa_atr"]]
        m = rest.merge(unrest, on="file", suffixes=("_rest", "_unrest"))
        m["delta_fa_atr"] = m["fa_atr_unrest"] - m["fa_atr_rest"]
        sub = sample_by_percentile(m, n)
        sub["class"] = label
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pos-feat-rest",   type=Path, required=True)
    ap.add_argument("--pos-feat-unrest", type=Path, required=True)
    ap.add_argument("--neg-feat-rest",   type=Path, required=True)
    ap.add_argument("--neg-feat-unrest", type=Path, required=True)
    ap.add_argument("--pos-pdb-rest",    type=Path, required=True)
    ap.add_argument("--pos-pdb-unrest",  type=Path, required=True)
    ap.add_argument("--neg-pdb-rest",    type=Path, required=True)
    ap.add_argument("--neg-pdb-unrest",  type=Path, required=True)
    ap.add_argument("--n-binders", type=int, default=5)
    ap.add_argument("--n-decoys",  type=int, default=5)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    sample = build_sample(args.pos_feat_rest, args.pos_feat_unrest,
                          args.neg_feat_rest, args.neg_feat_unrest,
                          args.n_binders, args.n_decoys)
    print(f"Sampled {len(sample)} structures "
          f"({(sample['class'] == 'binder').sum()} binders + "
          f"{(sample['class'] == 'decoy').sum()} decoys)", flush=True)

    rows = []
    for _, r in sample.iterrows():
        fname = r["file"]
        cls = r["class"]
        rest_dir   = args.pos_pdb_rest   if cls == "binder" else args.neg_pdb_rest
        unrest_dir = args.pos_pdb_unrest if cls == "binder" else args.neg_pdb_unrest
        for state, dpath in [("rest", rest_dir), ("unrest", unrest_dir)]:
            pdb = dpath / fname
            if not pdb.exists():
                print(f"  MISSING {state} {pdb}", flush=True)
                continue
            try:
                d = decompose_one(pdb)
            except Exception as e:
                print(f"  ERROR  {state} {fname}: {e}", flush=True)
                continue
            d["state"] = state
            d["class"] = cls
            d["sample_delta_fa_atr"] = float(r["delta_fa_atr"])
            rows.append(d)
            print(f"  OK  {state:6s} {cls:6s} {fname:55s} "
                  f"total={d['total_kcal']:+10.1f}  "
                  f"fa_atr={d['fa_atr_kcal']:+9.1f}  "
                  f"gbsa={d['gbsa_kcal']:+8.1f}", flush=True)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"\nWrote {args.output_csv}  ({len(df)} rows)", flush=True)

    # Summary: per-class mean delta (unrest - rest) for total + key components
    print("\nAggregate delta (unrest - rest, kcal/mol):")
    pivot_cols = ["total_kcal", "nonbonded_kcal", "fa_atr_kcal", "fa_rep_kcal",
                  "gbsa_kcal", "bond_kcal", "angle_kcal", "torsion_kcal"]
    for cls in df["class"].unique():
        sub = df[df["class"] == cls]
        rest = sub[sub.state == "rest"].set_index("file")
        unrest = sub[sub.state == "unrest"].set_index("file")
        common = sorted(set(rest.index) & set(unrest.index))
        if not common:
            continue
        d = unrest.loc[common, pivot_cols] - rest.loc[common, pivot_cols]
        print(f"  {cls} (N={len(common)}): " +
              "  ".join(f"{c.replace('_kcal','')}={d[c].mean():+.1f}" for c in pivot_cols))


if __name__ == "__main__":
    main()
