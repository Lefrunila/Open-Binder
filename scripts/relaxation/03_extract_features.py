#!/usr/bin/env python3
"""
03_extract_features.py

Extract 27 interface features from relaxed nanobody-antigen PDB files using
OpenMM, mdtraj, ProDy, BioPython, and sc_connolly (Connolly SES shape complementarity).

Features extracted (FEATURE_COLS):
    complex_normalized, dG_cross, dG_cross/dSASAx100, dSASA_hphobic,
    dSASA_int, dSASA_polar, delta_unsatHbonds, dslf_fa13, fa_atr,
    hbond_E_fraction, hbond_bb_sc, hbond_lr_bb, hbond_sc, hbond_sr_bb,
    hbonds_int, nres_int, omega, per_residue_energy_int, pro_close,
    rama_prepro, ref, side1_normalized, side1_score, side2_normalized,
    side2_score, yhh_planarity, sc

The iface-csv must have columns: file, vhh_chains, antigen_chains, label
  - file         : PDB filename (basename)
  - vhh_chains   : single letter or comma-separated chain IDs for the VHH
  - antigen_chains: single letter or comma-separated chain IDs for antigen
  - label        : 1 (binder) or 0 (non-binder)

Checkpoint at {output}.ckpt.jsonl enables safe resumption with --append.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import traceback
import warnings
from pathlib import Path

import multiprocessing as mp
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

import prody
from Bio.PDB import PDBParser, PPBuilder, calc_dihedral

from openmm import unit, VerletIntegrator, CustomNonbondedForce
from openmm.app import ForceField, Modeller, NoCutoff, Simulation

# sc_connolly: open-source Connolly SES shape complementarity (replaces MSMS)
_SC_DIR = str(Path(__file__).resolve().parent.parent / "v3")
if _SC_DIR not in sys.path:
    sys.path.insert(0, _SC_DIR)
from sc_connolly import compute_sc as _sc_connolly_compute_sc

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore")
prody.confProDy(verbosity="none")

# Use fork-based multiprocessing on Linux for OpenMM resource sharing
if sys.platform != "win32":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

# ── Feature columns (canonical order) ─────────────────────────────────────────
# v2: removed 5 redundant features (per_residue_energy_int = complex_normalized;
#     side1/2_score/normalized were fake proportional splits of dG_cross).
#     Added: nres_int_vhh, nres_int_ag, dSASA_vhh, dSASA_ag, hbond_density.
FEATURE_COLS = [
    "complex_normalized", "dG_cross", "dG_cross/dSASAx100", "dSASA_hphobic",
    "dSASA_int", "dSASA_polar", "delta_unsatHbonds", "dslf_fa13", "fa_atr",
    "hbond_E_fraction", "hbond_bb_sc", "hbond_lr_bb", "hbond_sc", "hbond_sr_bb",
    "hbonds_int", "nres_int", "omega", "pro_close",
    "rama_prepro", "ref", "yhh_planarity", "sc",
    "nres_int_vhh", "nres_int_ag", "dSASA_vhh", "dSASA_ag", "hbond_density",
]

# =============================================================================
# SHAPE COMPLEMENTARITY (Connolly SES, open-source)
# =============================================================================

# W_DECAY, DENSITY, _MSMS_BIN removed — sc_connolly handles SES internally


# _find_msms() removed — replaced by sc_connolly (open-source Connolly SES)


def _atoms_to_xyzr(pdb_path: str, chain_ids: list) -> np.ndarray:
    """Extract heavy-atom coordinates and VDW radii for atom parsing."""
    VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80, "H": 1.20}
    coords = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21] not in chain_ids:
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith("H"):
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            elem   = line[76:78].strip() or atom_name[0]
            radius = VDW.get(elem.upper(), 1.70)
            coords.append([x, y, z, radius])
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 4), dtype=np.float32)



def compute_sc(pdb_path: str, chain_a: list, chain_b: list) -> float | None:
    """Compute shape complementarity using sc_connolly (Connolly SES, open-source).

    Replaces the prior MSMS-based implementation. Uses the same Lawrence-Colman
    dot-product statistic but computed on a grid-based Connolly SES surface,
    achieving Pearson r=0.564, Spearman rho=0.611, MAE=0.025 vs Rosetta/MSMS
    on the full N=1,129 positive cohort.
    """
    vhh_chain = chain_a[0] if chain_a else "H"
    try:
        val = _sc_connolly_compute_sc(pdb_path, chain_vhh=vhh_chain, chain_ag="all")
        return float(val)
    except Exception:
        return None



# =============================================================================
# PHYSICS & STRUCTURAL FEATURE EXTRACTION
# =============================================================================

def _detect_disulfide_variants(topology, positions) -> list:
    """Return CYX variant labels for disulfide-bonded cysteine residues."""
    disulfide_residues: set[int] = set()
    for bond in topology.bonds():
        a1, a2 = bond[0], bond[1]
        if (
            a1.name == "SG" and a1.residue.name in ("CYS", "CYX")
            and a2.name == "SG" and a2.residue.name in ("CYS", "CYX")
        ):
            disulfide_residues.add(a1.residue.index)
            disulfide_residues.add(a2.residue.index)
    return [
        "CYX" if r.index in disulfide_residues else None
        for r in topology.residues()
    ]


def _fix_and_protonate(pdb_path):
    """Clean PDB and add hydrogens using PDBFixer at pH 7."""
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(pdb_path))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    ff       = ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(fixer.topology, fixer.positions)
    variants = _detect_disulfide_variants(fixer.topology, fixer.positions)
    modeller.addHydrogens(ff, pH=7.0, variants=variants)
    return modeller.positions, modeller.topology


def _energy_from_fixed(positions, topology, chain_ids_keep=None) -> float:
    """Compute OpenMM AMBER14+OBC2 potential energy in kcal/mol."""
    modeller = Modeller(topology, positions)
    if chain_ids_keep:
        to_del = [
            c for c in modeller.topology.chains()
            if c.id not in set(chain_ids_keep)
        ]
        modeller.delete(to_del)
    ff     = ForceField("amber14-all.xml", "implicit/obc2.xml")
    system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff)
    sim    = Simulation(
        modeller.topology, system, VerletIntegrator(0.001 * unit.picoseconds)
    )
    sim.context.setPositions(modeller.positions)
    return (
        sim.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )


def compute_lj_attractive(positions, topology) -> float:
    """Compute attractive Lennard-Jones term (fa_atr) in kcal/mol."""
    ff       = ForceField("amber14-all.xml", "implicit/obc2.xml")
    modeller = Modeller(topology, positions)
    system   = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff)

    nb = next(f for f in system.getForces() if f.__class__.__name__ == "NonbondedForce")
    cf = CustomNonbondedForce(
        "min(0, epsilon12*((sigma12/r)^12 - 2*(sigma12/r)^6));"
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

    cf.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    # Mirror all exclusions to keep context stable
    for i in range(nb.getNumExceptions()):
        p1, p2, *_ = nb.getExceptionParameters(i)
        cf.addExclusion(p1, p2)

    cf.setForceGroup(31)
    system.addForce(cf)

    sim = Simulation(
        modeller.topology, system, VerletIntegrator(0.001 * unit.picoseconds)
    )
    sim.context.setPositions(modeller.positions)
    return (
        sim.context.getState(getEnergy=True, groups={31})
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )


def compute_sasa_features(traj, vhh_chains: list, ag_chains: list) -> dict:
    """Element-masked SASA burial features."""
    import mdtraj as md

    top    = traj.topology
    v_idx  = [a.index for a in top.atoms if a.residue.chain.chain_id in vhh_chains]
    a_idx  = [a.index for a in top.atoms if a.residue.chain.chain_id in ag_chains]

    s_comp = md.shrake_rupley(traj, mode="atom")[0] * 100.0
    d_v    = md.shrake_rupley(traj.atom_slice(v_idx), mode="atom")[0] * 100.0 - s_comp[v_idx]
    d_a    = md.shrake_rupley(traj.atom_slice(a_idx), mode="atom")[0] * 100.0 - s_comp[a_idx]

    v_pol  = np.array([top.atom(i).element.symbol in ("N", "O") for i in v_idx], dtype=bool)
    a_pol  = np.array([top.atom(i).element.symbol in ("N", "O") for i in a_idx], dtype=bool)
    v_hph  = np.array([top.atom(i).element.symbol in ("C", "S") for i in v_idx], dtype=bool)
    a_hph  = np.array([top.atom(i).element.symbol in ("C", "S") for i in a_idx], dtype=bool)

    v_int  = {top.atom(v_idx[i]).residue.index for i, d in enumerate(d_v) if d > 1.0}
    a_int  = {top.atom(a_idx[i]).residue.index for i, d in enumerate(d_a) if d > 1.0}

    b_pol: set[int] = set()
    for i, atom_idx in enumerate(v_idx):
        at = top.atom(atom_idx)
        if d_v[i] > 0.5 and at.element.symbol in ("N", "O") and not at.name.startswith("H"):
            b_pol.add(atom_idx)
    for i, atom_idx in enumerate(a_idx):
        at = top.atom(atom_idx)
        if d_a[i] > 0.5 and at.element.symbol in ("N", "O") and not at.name.startswith("H"):
            b_pol.add(atom_idx)

    return {
        "dSASA_int":     float(d_v.sum() + d_a.sum()),
        "dSASA_polar":   float(d_v[v_pol].sum() + d_a[a_pol].sum()),
        "dSASA_hphobic": float(d_v[v_hph].sum() + d_a[a_hph].sum()),
        "nres_int":      len(v_int) + len(a_int),
        "nres_int_vhh":  len(v_int),
        "nres_int_ag":   len(a_int),
        "dSASA_vhh":     float(d_v.sum()),
        "dSASA_ag":      float(d_a.sum()),
        "_v_int":        v_int,
        "_a_int":        a_int,
        "_b_pol":        b_pol,
    }


def compute_hbond_features(
    traj, vhh_chains: list, ag_chains: list,
    v_int: set, a_int: set, b_pol: set,
) -> dict:
    """Decomposed H-bond network across the interface."""
    import mdtraj as md

    top        = traj.topology
    xyz        = traj.xyz[0] * 10.0          # nm → Å
    v_set      = set(vhh_chains)
    all_chains = set(vhh_chains) | set(ag_chains)

    polar = [
        a for a in top.atoms
        if a.residue.chain.chain_id in all_chains
        and a.element and a.element.symbol in ("N", "O")
        and not a.name.startswith("H")
    ]

    if not polar:
        return {k: 0 for k in [
            "hbonds_int", "hbond_E_fraction", "hbond_bb_sc",
            "hbond_lr_bb", "hbond_sr_bb", "hbond_sc", "delta_unsatHbonds",
        ]}

    pidx  = np.array([a.index for a in polar])
    dists = cdist(xyz[pidx], xyz[pidx])
    np.fill_diagonal(dists, 99.0)
    pi, pj = np.where((dists >= 2.5) & (dists < 3.5))
    ut = pi < pj
    pi, pj = pi[ut], pj[ut]

    n_bb_sc = n_lr_bb = n_sr_bb = n_sc = 0
    for i, j in zip(pi, pj):
        a1, a2   = polar[i], polar[j]
        bb1, bb2 = a1.name in ("N", "O"), a2.name in ("N", "O")
        seq = (
            abs(a1.residue.index - a2.residue.index)
            if a1.residue.chain == a2.residue.chain
            else 999
        )
        if bb1 and bb2:
            if seq <= 1 or {a1.name, a2.name} != {"N", "O"}:
                continue
            if seq < 5:
                n_sr_bb += 1
            else:
                n_lr_bb += 1
        elif bb1 or bb2:
            if {a1.element.symbol, a2.element.symbol} == {"N", "O"}:
                n_bb_sc += 1
        else:
            n_sc += 1

    n_cross = n_tot_bh = 0
    bh_atoms: set[int] = set()
    try:
        bh = md.baker_hubbard(traj, periodic=False)
        for d, h, a in bh:
            c1 = top.atom(d).residue.chain.chain_id
            c2 = top.atom(a).residue.chain.chain_id
            if c1 not in all_chains or c2 not in all_chains:
                continue
            bh_atoms.update([d, a])
            n_tot_bh += 1
            if (c1 in v_set) != (c2 in v_set):
                n_cross += 1
    except Exception:
        pass

    return {
        "hbonds_int":         n_cross,
        "hbond_E_fraction":   n_cross / max(n_tot_bh, 1),
        "hbond_bb_sc":       -n_bb_sc,
        "hbond_lr_bb":       -n_lr_bb,
        "hbond_sr_bb":       -n_sr_bb,
        "hbond_sc":          -n_sc,
        "delta_unsatHbonds":  len(b_pol - bh_atoms),
    }


def compute_geometry(pdb_path, chains: set) -> dict:
    """Backbone geometry: omega deviation, Ramachandran outliers, proline pucker,
    tyrosine planarity, disulfide bonds, and residue count."""
    struct = prody.parsePDB(str(pdb_path))
    bio    = PDBParser(QUIET=True).get_structure("s", str(pdb_path))

    omega_devs: list[float] = []
    rama   = 0
    pro_devs: list[float] = []
    ref_tot = 0
    yhh_sum = 0.0

    for model in bio:
        for chain in model:
            if chain.id not in chains:
                continue
            residues = [r for r in chain if "CA" in r and "C" in r and "N" in r]
            for i in range(1, len(residues)):
                try:
                    w = calc_dihedral(
                        residues[i - 1]["CA"].get_vector(),
                        residues[i - 1]["C"].get_vector(),
                        residues[i]["N"].get_vector(),
                        residues[i]["CA"].get_vector(),
                    )
                    omega_devs.append(abs(180.0 - abs(np.degrees(w))))
                except Exception:
                    pass

            for pp in PPBuilder().build_peptides(chain):
                for ph, ps in pp.get_phi_psi_list():
                    if ph and ps:
                        ph_d, ps_d = np.degrees(ph), np.degrees(ps)
                        in_allowed = (
                            (-180 <= ph_d <= -40 and -100 <= ps_d <= 50)
                            or (-180 <= ph_d <= -40 and 100 <= ps_d <= 180)
                            or (40 <= ph_d <= 80 and 20 <= ps_d <= 80)
                        )
                        if not in_allowed:
                            rama += 1

            for r in chain:
                if r.get_resname() == "TYR":
                    try:
                        ang = calc_dihedral(
                            r["CE1"].get_vector(),
                            r["CZ"].get_vector(),
                            r["OH"].get_vector(),
                            r["HH"].get_vector(),
                        )
                        yhh_sum += float(np.sin(ang) ** 2)
                    except Exception:
                        pass

    for cid in chains:
        pro = struct.select(f"chain {cid} and resname PRO")
        if pro:
            for rnum in set(pro.getResnums()):
                n_sel  = pro.select(f"resnum {rnum} and name N")
                cd_sel = pro.select(f"resnum {rnum} and name CD")
                if n_sel and cd_sel:
                    pro_devs.append(np.linalg.norm(n_sel.getCoords()[0] - cd_sel.getCoords()[0]))
        ref_tot += len(struct.select(f"chain {cid} and protein and name CA") or [])

    # Disulfide count
    dslf = 0.0
    all_chain_sel = " or ".join(f"chain {cid}" for cid in chains)
    sg = struct.select(f"({all_chain_sel}) and resname CYS and name SG")
    if sg and len(sg) >= 2:
        coords = sg.getCoords()
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                if np.linalg.norm(coords[i] - coords[j]) < 2.1:
                    dslf -= 1.0

    return {
        "omega":       sum(np.sin(np.radians(d)) ** 2 * 30.0 for d in omega_devs),
        "rama_prepro": float(rama),
        "pro_close":   sum(150.0 * (d - 1.477) ** 2 for d in pro_devs),
        "yhh_planarity": float(yhh_sum),
        "dslf_fa13":   float(dslf),
        "ref":         float(ref_tot),
    }


# =============================================================================
# Main extraction wrapper
# =============================================================================

def extract_features(pdb_path: str, v_chains: list, a_chains: list, label) -> dict:
    """
    Extract all 27 features for a single PDB file.

    Returns a dict with keys: file, label, status, + all FEATURE_COLS.
    """
    import mdtraj as md

    res = {
        "file":   Path(pdb_path).name,
        "label":  label,
        "status": "ERROR",
    }
    for f in FEATURE_COLS:
        res[f] = None

    try:
        traj = md.load(str(pdb_path))
        sasa = compute_sasa_features(traj, v_chains, a_chains)
        hb   = compute_hbond_features(
            traj, v_chains, a_chains,
            sasa["_v_int"], sasa["_a_int"], sasa["_b_pol"],
        )
        geo  = compute_geometry(pdb_path, set(v_chains) | set(a_chains))

        pos, top = _fix_and_protonate(pdb_path)
        e_complex = _energy_from_fixed(pos, top)
        e_vhh     = _energy_from_fixed(pos, top, v_chains)
        e_ag      = _energy_from_fixed(pos, top, a_chains)
        dg        = e_complex - e_vhh - e_ag

        res.update({k: v for k, v in sasa.items() if not k.startswith("_")})
        res.update(hb)
        res.update(geo)

        n_int   = max(res["nres_int"], 1)
        dsasa   = max(res["dSASA_int"], 1.0)

        res.update({
            "dG_cross":              dg,
            "fa_atr":                compute_lj_attractive(pos, top),
            "complex_normalized":    dg / n_int,
            "dG_cross/dSASAx100":   dg / dsasa * 100,
            "hbond_density":         res["hbonds_int"] / dsasa,
            "sc":                    compute_sc(str(pdb_path), v_chains, a_chains),
            "status":                "OK",
        })

    except Exception as e:
        res["status"] = f"ERROR: {e}"
        traceback.print_exc()

    return res


def _worker(job: tuple) -> dict:
    return extract_features(*job)


# =============================================================================
# Main
# =============================================================================

# Suffix map: mode -> (PDB-dir suffix, output-stem suffix)
# The PDB-dir suffix is appended to the inferred set name ("positives" /
# "negatives") derived from the iface-csv filename.  The output-stem suffix
# is inserted before the file extension of --output, so we can write the
# restrained and unrestrained CSVs side-by-side without collision.
_MODE_DIR_SUFFIX = {
    "restrained":    "_relaxed",
    "unrestrained":  "_unrest",
}
_MODE_OUT_SUFFIX = {
    "restrained":    "",
    "unrestrained":  "_unrest",
}


def _infer_set_from_iface(iface_csv: Path) -> str:
    """Infer dataset ('positives' or 'negatives') from iface-csv filename."""
    stem = iface_csv.stem.lower()
    if "positive" in stem:
        return "positives"
    if "negative" in stem:
        return "negatives"
    raise ValueError(
        f"Cannot infer positives/negatives from iface-csv name '{iface_csv.name}'. "
        f"Expected 'positive' or 'negative' in stem. "
        f"Pass --relaxed <dir> explicitly to override."
    )


def _resolve_mode_paths(
    mode: str,
    iface_csv: Path,
    base_output: Path,
    relaxed_override: Path | None,
) -> tuple[Path, Path]:
    """
    Resolve (pdb_dir, output_csv) for a single mode.

    Rules:
      - relaxed_override, if given, wins for the pdb_dir (with a warning).
      - pdb_dir is otherwise DATA_DIR / f"{set_name}{_MODE_DIR_SUFFIX[mode]}".
      - output_csv is base_output with the mode output-stem suffix inserted
        before the extension.  For mode=='restrained' the output path is
        identical to base_output (preserves today's behavior).
    """
    if relaxed_override is not None:
        print(
            f"[warn] --relaxed {relaxed_override} overrides --mode {mode} "
            f"input directory.",
            flush=True,
        )
        pdb_dir = relaxed_override
    else:
        set_name = _infer_set_from_iface(iface_csv)
        pdb_dir  = DATA_DIR / f"{set_name}{_MODE_DIR_SUFFIX[mode]}"

    out_suffix = _MODE_OUT_SUFFIX[mode]
    if out_suffix:
        output_csv = base_output.with_name(
            base_output.stem + out_suffix + base_output.suffix
        )
    else:
        output_csv = base_output

    return pdb_dir, output_csv


def _run_extraction(
    mode: str,
    pdb_dir: Path,
    iface_csv: Path,
    output_csv: Path,
    nproc: int,
    append: bool,
) -> None:
    """Run the feature-extraction loop for a single mode/dir/output combo."""
    import json

    ckpt_file = Path(str(output_csv) + ".ckpt.jsonl")

    print(f"\n=== mode={mode} ===", flush=True)
    print(f"PDB dir          : {pdb_dir}", flush=True)
    print(f"Output CSV       : {output_csv}", flush=True)
    print(f"Checkpoint file  : {ckpt_file}", flush=True)

    if not pdb_dir.is_dir():
        print(
            f"[warn] PDB dir does not exist: {pdb_dir} — skipping mode '{mode}'.",
            flush=True,
        )
        return

    # Load interface metadata
    meta: dict[str, dict] = {}
    with open(iface_csv) as fh:
        for row in csv.DictReader(fh):
            vhh_chains = [c.strip() for c in row["vhh_chains"].split(",")]
            ag_chains  = [c.strip() for c in row["antigen_chains"].split(",")]
            meta[row["file"]] = {
                "v": vhh_chains,
                "a": ag_chains,
                "l": int(row["label"]),
            }

    # Collect already-done files (from checkpoint or existing output CSV).
    # Each mode owns its own checkpoint file at <output_csv>.ckpt.jsonl.
    done: set[str] = set()
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "OK":
                    done.add(r["file"])
            except Exception:
                pass
    if append and output_csv.exists():
        with open(output_csv) as fh:
            for row in csv.DictReader(fh):
                done.add(row["file"])

    # Build job list
    jobs = []
    for pdb in sorted(pdb_dir.glob("*.pdb")):
        if pdb.name in meta and pdb.name not in done:
            m = meta[pdb.name]
            jobs.append((str(pdb), m["v"], m["a"], m["l"]))

    print(f"Metadata entries : {len(meta)}", flush=True)
    print(f"Already done     : {len(done)}", flush=True)
    print(f"Jobs to run      : {len(jobs)}", flush=True)

    if not jobs:
        print("Nothing to do for this mode!", flush=True)
        return

    # Write header only if creating a new file
    write_header = not (append and output_csv.exists())
    fieldnames   = ["file", "label", "status"] + FEATURE_COLS
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pool = mp.Pool(nproc)
    try:
        with (
            open(output_csv, "a" if append else "w", newline="") as out_fh,
            open(ckpt_file, "a") as ckpt_fh,
        ):
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()

            for res in tqdm(
                pool.imap_unordered(_worker, jobs),
                total=len(jobs),
                desc=f"Extracting features [{mode}]",
            ):
                writer.writerow(res)
                out_fh.flush()

                ckpt_entry = {"file": res["file"], "status": res["status"]}
                json.dump(ckpt_entry, ckpt_fh)
                ckpt_fh.write("\n")
                ckpt_fh.flush()
    finally:
        pool.terminate()
        pool.join()

    print(f"\nOutput : {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["restrained", "unrestrained", "both"],
        default="restrained",
        help=(
            "Which minimized-PDB set to extract features from. "
            "'restrained' reads data/{positives,negatives}_relaxed/ and writes "
            "the default --output path (preserves legacy behavior). "
            "'unrestrained' reads data/{positives,negatives}_unrest/ and writes "
            "a sibling CSV with '_unrest' inserted into the output stem. "
            "'both' runs each sequentially to two distinct CSVs. "
            "Default: restrained."
        ),
    )
    parser.add_argument(
        "--relaxed",
        type=Path,
        default=None,
        help=(
            "Explicit directory of relaxed PDB files. "
            "If passed, overrides the --mode-derived input directory "
            "(a warning is printed). In 'both' mode the override applies to "
            "every mode run, which is rarely what you want — prefer running "
            "each mode separately when using --relaxed."
        ),
    )
    parser.add_argument(
        "--iface-csv",
        type=Path,
        required=True,
        help="CSV with columns: file, vhh_chains, antigen_chains, label",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "features_v27.csv",
        help=(
            "Output CSV path (default: ../data/features_v27.csv). "
            "In 'unrestrained'/'both' modes, the unrestrained CSV is written "
            "to the same directory with '_unrest' inserted before the extension."
        ),
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="Number of parallel workers.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output CSV instead of overwriting.",
    )
    args = parser.parse_args()

    modes = ["restrained", "unrestrained"] if args.mode == "both" else [args.mode]

    for m in modes:
        pdb_dir, output_csv = _resolve_mode_paths(
            mode=m,
            iface_csv=args.iface_csv,
            base_output=args.output,
            relaxed_override=args.relaxed,
        )
        _run_extraction(
            mode=m,
            pdb_dir=pdb_dir,
            iface_csv=args.iface_csv,
            output_csv=output_csv,
            nproc=args.nproc,
            append=args.append,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
