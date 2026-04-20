#!/usr/bin/env python3
"""
02_relax_no_restraints.py

Identical to 02_relax.py but WITHOUT CA harmonic restraints.
Used for structures where the placed VHH has steric clashes that require
free energy minimization to resolve (VHH must be able to move away from antigen).

Usage:
    conda run -n openmm_cuda python scripts/02_relax_no_restraints.py \
        --input  data/negatives \
        --output data/negatives_relaxed \
        --files  data/negatives/.no_ca_restraints.txt \
        --platform cuda
"""

import argparse
import json
import sys
import time
import traceback
from io import StringIO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

K_RESTRAINT_KJ   = 10.0
PH               = 7.0
MAX_ITER         = 2000
ENERGY_TOL_KJ    = 1.0

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "CYX", "ACE", "NME",
}


def _log(fname, step, t0, extra=""):
    elapsed = time.time() - t0
    print(f"  [{fname[:45]:<45s}] {step:<30s} {elapsed:6.1f}s {extra}", flush=True)


def prepare_one_no_restraints(pdb_in: str, pdb_out: str, platform_spec: str) -> dict:
    """Full PDBFixer + createSystem prep, but WITHOUT CA restraints."""
    fname  = Path(pdb_in).name
    result = {"file": fname, "pdb_out": pdb_out,
              "platform_spec": platform_spec, "status": "ERROR", "msg": ""}
    t0 = time.time()

    try:
        import openmm.app as app
        import openmm as mm
        import openmm.unit as unit
        from pdbfixer import PDBFixer

        raw_lines        = Path(pdb_in).read_text().splitlines()
        protein_lines    = []
        nonprotein_lines = []
        for line in raw_lines:
            rec = line[:6].strip()
            if rec in ("ATOM", "TER"):
                if rec == "ATOM" and len(line) > 20:
                    resname = line[17:20].strip()
                    if resname == "UNK":
                        continue
                    # Ligand/cofactor deposited as ATOM (e.g. GDP) — route to
                    # non-protein (dropped by removeHeterogens), promote record
                    # type to HETATM for downstream consistency.
                    if resname not in PROTEIN_RESIDUES and resname != "PCA":
                        nonprotein_lines.append("HETATM" + line[6:])
                        continue
                protein_lines.append(line)
            elif rec == "HETATM":
                res = line[17:20].strip()
                if res not in ("HOH", "WAT", "H2O"):
                    nonprotein_lines.append(line)
            elif rec in ("SSBOND", "LINK", "END"):
                protein_lines.append(line)
        _log(fname, "parse", t0, f"{len(protein_lines)} protein lines")

        # Chain sort + renumber
        chains_atoms: dict = {}
        chain_order: list  = []
        for line in protein_lines:
            if line[:4] in ("ATOM", "HETA") and len(line) > 26:
                ch = line[21]
                if ch not in chains_atoms:
                    chains_atoms[ch] = []
                    chain_order.append(ch)
                chains_atoms[ch].append(line)

        renumbered = []
        for ch in chain_order:
            cur_orig_res    = None
            new_res_counter = 0
            for line in chains_atoms[ch]:
                try:
                    orig_res = int(line[22:26])
                except (ValueError, IndexError):
                    renumbered.append(line)
                    continue
                if orig_res != cur_orig_res:
                    cur_orig_res = orig_res
                    new_res_counter += 1
                line = line[:22] + f"{new_res_counter:4d}" + line[26:]
                renumbered.append(line)
            renumbered.append("TER")
        protein_lines = renumbered

        # PCA → GLN
        fixed_lines = []
        for line in protein_lines:
            if line[:4] in ("ATOM", "HETA") and len(line) > 20 and line[17:20].strip() == "PCA":
                line = line[:17] + "GLN" + line[20:]
                atom_name = line[12:16].strip()
                if atom_name == "OE":
                    line = line[:12] + " OE1" + line[16:]
                if line[12:16].strip() not in ("N", "CA", "C", "O", "CB", "CG", "CD", "OE1"):
                    continue
            fixed_lines.append(line)
        protein_lines = fixed_lines

        # PDBFixer
        fixer = PDBFixer(pdbfile=StringIO("\n".join(protein_lines)))
        _log(fname, "fixer_init", t0)
        fixer.findMissingResidues()

        # Don't fill large gaps (>5 residues) in antigen chains — PDBFixer
        # builds unphysical geometry that can split chains into phantom
        # segments (e.g. 7DSS_1 chain 3 → chains A + B via large-gap bridging).
        _chains_list = list(fixer.topology.chains())
        _vhh_indices = {i for i, ch in enumerate(_chains_list) if ch.id == "H"}
        _before = len(fixer.missingResidues)
        fixer.missingResidues = {
            k: v for k, v in fixer.missingResidues.items()
            if k[0] in _vhh_indices or len(v) <= 5
        }
        if len(fixer.missingResidues) < _before:
            _log(fname, "large_gap_filter", t0,
                 f"dropped {_before - len(fixer.missingResidues)} large antigen gaps")

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.removeHeterogens(keepWater=False)
        fixer.addMissingHydrogens(PH)
        n_atoms = fixer.topology.getNumAtoms()
        _log(fname, "addMissingHydrogens", t0, f"{n_atoms} atoms")

        modeller = app.Modeller(fixer.topology, fixer.positions)

        # Strip H2/H3 on non-N-terminal residues and SH on disulfide CYS
        bad_atoms = []
        for chain in modeller.topology.chains():
            residues = list(chain.residues())
            bonded_sg = set()
            for bond in modeller.topology.bonds():
                a1, a2 = bond[0], bond[1]
                if (a1.residue.name in ("CYS","CYX") and a1.name == "SG" and
                    a2.residue.name in ("CYS","CYX") and a2.name == "SG"):
                    bonded_sg.add(a1.index)
                    bonded_sg.add(a2.index)
            for i, res in enumerate(residues):
                for atom in res.atoms():
                    if atom.name in ("H2", "H3") and i != 0:
                        bad_atoms.append(atom)
                    if res.name in ("CYS","CYX") and atom.name == "HG":
                        sg = next((a for a in res.atoms() if a.name == "SG"), None)
                        if sg and sg.index in bonded_sg:
                            bad_atoms.append(atom)
        if bad_atoms:
            modeller.delete(bad_atoms)
            _log(fname, f"stripped {len(bad_atoms)} terminal-artifact atoms", t0)

        # createSystem
        ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")

        def _create_system(top):
            return ff.createSystem(
                top,
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                rigidWater=True,
            )

        try:
            system = _create_system(modeller.topology)
            _log(fname, "createSystem", t0, f"{system.getNumParticles()} particles")
        except Exception as e1:
            _log(fname, "createSystem FAILED — retrying with terminal fix", t0, str(e1)[:60])
            fixer.findMissingAtoms()
            fixer.addMissingAtoms(seed=0)
            modeller = app.Modeller(fixer.topology, fixer.positions)
            mid_oxt = [
                a for ch in modeller.topology.chains()
                for i, r in enumerate(list(ch.residues()))
                for a in r.atoms()
                if i < len(list(ch.residues())) - 1
                and a.name in ("OXT", "OC1", "OC2")
            ]
            if mid_oxt:
                modeller.delete(mid_oxt)
            bad_atoms2 = []
            for chain in modeller.topology.chains():
                residues2 = list(chain.residues())
                for i, res in enumerate(residues2):
                    for atom in res.atoms():
                        if atom.name in ("H2", "H3") and i != 0:
                            bad_atoms2.append(atom)
            if bad_atoms2:
                modeller.delete(bad_atoms2)
            try:
                system = _create_system(modeller.topology)
                result["msg"] = f"terminal fix applied: {e1}"
            except Exception as e2:
                _log(fname, "createSystem FAILED again — dropping N-term GLN", t0, str(e2)[:60])
                nterm_gln = [
                    a for ch in modeller.topology.chains()
                    for i, res in enumerate(list(ch.residues()))
                    for a in res.atoms()
                    if i == 0 and res.name == "GLN"
                ]
                if nterm_gln:
                    modeller.delete(nterm_gln)
                system = _create_system(modeller.topology)
                result["msg"] = f"terminal fix + drop N-term GLN: {e1}"
                _log(fname, "createSystem (after GLN drop)", t0, f"{system.getNumParticles()} particles")

        # ── NO CA RESTRAINTS (intentional) ──────────────────────────────────────
        _log(fname, "NO_CA_RESTRAINTS (free minimization)", t0, "clash-resolution mode")

        pos_nm       = modeller.positions.value_in_unit(unit.nanometer)
        positions_nm = [[x, y, z] for x, y, z in pos_nm]
        system_xml   = mm.XmlSerializer.serialize(system)

        orig_chains = []
        for line in protein_lines:
            if line[:4] == "ATOM" and len(line) > 21 and line[21].strip():
                ch = line[21]
                if ch not in orig_chains:
                    orig_chains.append(ch)

        # Assign original chain IDs onto the topology BEFORE writing so
        # PDBFile.writeFile + keepIds preserves them, bypassing OpenMM's
        # alphabetical-rename behavior for non-A-first chain orderings.
        _topo_chains = list(modeller.topology.chains())
        if len(_topo_chains) == len(orig_chains):
            for _tc, _orig_id in zip(_topo_chains, orig_chains):
                _tc.id = _orig_id

        topo_buf = StringIO()
        app.PDBFile.writeFile(modeller.topology, modeller.positions, topo_buf,
                              keepIds=True)
        topology_pdb = topo_buf.getvalue()

        _log(fname, "serialise  PREP_TOTAL", t0, f"xml={len(system_xml)//1024}KB")

        result.update({
            "status":           "READY",
            "system_xml":       system_xml,
            "positions_nm":     positions_nm,
            "topology_pdb":     topology_pdb,
            "orig_chains":      orig_chains,
            "nonprotein_lines": nonprotein_lines,
            "n_restrained":     0,
        })

    except Exception as e:
        result["msg"] = str(e)
        _log(fname, f"PREP-ERR: {e}", t0)
        traceback.print_exc()

    return result


def minimize_one(prep: dict) -> dict:
    """GPU minimization — identical to 02_relax.py's _minimize_one."""
    fname         = prep["file"]
    pdb_out       = prep["pdb_out"]
    platform_spec = prep["platform_spec"]
    result        = {"file": fname, "status": "ERROR", "msg": prep.get("msg", "")}
    t0 = time.time()

    try:
        import openmm.app as app
        import openmm as mm
        import openmm.unit as unit

        system    = mm.XmlSerializer.deserialize(prep["system_xml"])
        positions = [mm.Vec3(x, y, z) * unit.nanometer for x, y, z in prep["positions_nm"]]
        topo_pdb  = app.PDBFile(StringIO(prep["topology_pdb"]))

        plat_name, plat_props = platform_spec
        platform = mm.Platform.getPlatformByName(plat_name)

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.004 * unit.picoseconds
        )
        sim = app.Simulation(topo_pdb.topology, system, integrator,
                             platform, plat_props)
        sim.context.setPositions(positions)
        _log(fname, "sim_setup", t0, f"platform={plat_name}")

        sim.minimizeEnergy(
            tolerance=ENERGY_TOL_KJ * unit.kilojoules_per_mole,
            maxIterations=MAX_ITER,
        )
        _log(fname, "minimizeEnergy", t0)

        state     = sim.context.getState(getPositions=True, getEnergy=True)
        e_after   = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        minimized_pos = state.getPositions()

        # Defensively re-assign original chain IDs onto the reloaded topology
        # in case PDBFile(StringIO) re-alphabetized them. Combined with
        # keepIds=True this bypasses OpenMM's alphabetical-rename behavior.
        orig_chains = prep["orig_chains"]
        _topo_chains = list(topo_pdb.topology.chains())
        if len(_topo_chains) == len(orig_chains):
            for _tc, _orig_id in zip(_topo_chains, orig_chains):
                _tc.id = _orig_id

        # Render PDB to string first so we can audit chain topology before writing
        protein_buf = StringIO()
        app.PDBFile.writeFile(topo_pdb.topology, minimized_pos, protein_buf,
                              keepIds=True)
        protein_pdb = protein_buf.getvalue()

        omm_chains: list[str] = []
        for line in protein_pdb.splitlines():
            if line[:4] == "ATOM" and len(line) > 21:
                ch = line[21]
                if ch not in omm_chains:
                    omm_chains.append(ch)

        # Strict chain-count check: phantom chains silently produced garbage
        # features before (dG_cross ~ -11789 for 7DSS_1). Fail loudly instead.
        if len(omm_chains) != len(orig_chains):
            raise ValueError(
                f"chain count mismatch after relax: omm={omm_chains} orig={orig_chains}"
            )

        if omm_chains != orig_chains:
            # Fallback: positional zip remap if explicit assignment failed
            ch_map = dict(zip(omm_chains, orig_chains))
            remapped = []
            for line in protein_pdb.splitlines():
                if line[:4] in ("ATOM", "HETA", "TER ") and len(line) > 21:
                    line = line[:21] + ch_map.get(line[21], line[21]) + line[22:]
                remapped.append(line)
            protein_pdb = "\n".join(remapped)

        pdb_out_path = Path(pdb_out)
        pdb_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdb_out_path, "w") as fh:
            fh.write(
                f"REMARK  OpenMM AMBER14+OBC2 UNRESTRAINED minimization  "
                f"E_after={e_after:.1f} kJ/mol  max_iter={MAX_ITER}\n"
            )
            for line in protein_pdb.splitlines():
                if line.startswith(("ATOM", "TER")):
                    fh.write(line + "\n")
            # Append non-protein (ligands, cofactors) preserved from input
            for l in prep["nonprotein_lines"]:
                fh.write(l + "\n")
            fh.write("END\n")

        _log(fname, "write_pdb  TOTAL", t0, pdb_out)
        result.update({"status": "OK", "msg": prep.get("msg", ""), "e_after": e_after})

    except Exception as e:
        result["msg"] = str(e)
        _log(fname, f"MIN-ERR: {e}", t0)
        traceback.print_exc()

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",    type=Path, default=DATA_DIR / "negatives")
    parser.add_argument("--output",   type=Path, default=DATA_DIR / "negatives_relaxed")
    parser.add_argument("--files",    type=Path,
                        help="Text file listing PDB filenames to process (one per line)")
    parser.add_argument("--platform", default="cuda",
                        choices=["cuda", "opencl", "cpu"])
    args = parser.parse_args()

    if args.platform == "cuda":
        platform_spec = ("CUDA", {"CudaPrecision": "mixed"})
    elif args.platform == "opencl":
        platform_spec = ("OpenCL", {})
    else:
        platform_spec = ("CPU", {})

    # Collect files
    if args.files and args.files.exists():
        pdbs = [args.input / l.strip() for l in args.files.read_text().splitlines()
                if l.strip() and (args.input / l.strip()).exists()]
    else:
        pdbs = sorted(args.input.glob("*.pdb"))

    ckpt_file = args.output / ".relax_ckpt.jsonl"
    relaxed_done = set()
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "OK":
                    relaxed_done.add(r["file"])
            except Exception:
                pass

    jobs = [(str(p), str(args.output / p.name), platform_spec)
            for p in pdbs if p.name not in relaxed_done]

    print(f"Files to relax (no restraints): {len(jobs)}")
    if not jobs:
        print("Nothing to do.")
        return

    n_ok = n_err = 0
    for pdb_in, pdb_out, plat in jobs:
        print(f"\n--- {Path(pdb_in).name} ---")
        prep = prepare_one_no_restraints(pdb_in, pdb_out, plat)
        if prep["status"] != "READY":
            print(f"  PREP FAILED: {prep['msg']}")
            n_err += 1
            ckpt_file.parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_file, "a") as ck:
                ck.write(json.dumps({"file": prep["file"], "status": "ERROR",
                                     "msg": prep["msg"]}) + "\n")
            continue
        res = minimize_one(prep)
        status = res["status"]
        if status == "OK":
            n_ok += 1
            print(f"  OK: {pdb_out}")
        else:
            n_err += 1
            print(f"  ERR: {res['msg']}")
        with open(ckpt_file, "a") as ck:
            ck.write(json.dumps({"file": res["file"], "status": status,
                                 "msg": res.get("msg", "")}) + "\n")

    print(f"\nDone: {n_ok} OK, {n_err} errors")


if __name__ == "__main__":
    main()
