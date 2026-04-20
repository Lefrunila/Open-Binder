#!/usr/bin/env python3
"""
02_relax.py

OpenMM AMBER14 + OBC2 implicit-solvent relaxation with CA harmonic restraints.

Two-stage pipeline:
  CPU pool  → PDBFixer + createSystem + CA restraints  ~4-5 s each
  GPU pool  → Simulation setup + L-BFGS minimize       ~1-3 s each

Checkpoint file at {output_dir}/.relax_ckpt.jsonl enables safe resumption.
"""

import argparse
import json
import math
import multiprocessing as mp
import sys
import threading
import time
import traceback
from io import StringIO
from pathlib import Path

import numpy as np

# ── Relaxation hyper-parameters ───────────────────────────────────────────────
K_RESTRAINT_KJ = 10.0   # kJ/mol/nm²
MAX_ITER       = 2000
ENERGY_TOL_KJ  = 1.0
PH             = 7.0

PROTEIN_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIE", "HID", "HIP", "CYX", "ACE", "NME",
}

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR.parent / "data"

if sys.platform != "win32":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass


class _Tee:
    """Write to two streams simultaneously."""
    def __init__(self, a, b):
        self.a, self.b = a, b
    def write(self, data):
        self.a.write(data)
        self.b.write(data)
    def flush(self):
        self.a.flush()
        self.b.flush()
    def fileno(self):
        return self.a.fileno()


def _log(fname, step, t0, extra=""):
    elapsed = time.time() - t0
    print(f"  [{fname[:45]:<45s}] {step:<30s} {elapsed:6.1f}s {extra}", flush=True)


# =============================================================================
# Stage 1 – CPU worker
# =============================================================================

def _prepare_one(job: tuple) -> dict:
    pdb_in, pdb_out, platform_spec = job
    fname  = Path(pdb_in).name
    result = {"file": fname, "pdb_out": pdb_out,
              "platform_spec": platform_spec, "status": "ERROR", "msg": ""}
    t0 = time.time()

    # ── Load from pre-computed cache if available ─────────────────────────────
    import json as _json
    prep_dir   = Path(pdb_out).parent / ".prep"
    cache_json = prep_dir / (fname + ".json")
    if cache_json.exists():
        try:
            # Invalidate stale cache: if input PDB is newer than cache, re-prep
            pdb_mtime = Path(pdb_in).stat().st_mtime
            cache_mtime = cache_json.stat().st_mtime
            if pdb_mtime > cache_mtime:
                _log(fname, "cache stale (PDB newer), re-computing", t0)
                raise ValueError("stale cache")
            payload = _json.loads(cache_json.read_text())
            xml_path = Path(payload.get("system_xml_path", ""))
            if xml_path.exists():
                # Leave system_xml unloaded — GPU worker loads it just before use
                # so large XMLs don't accumulate in the imap queue
                payload["system_xml"] = None
                _log(fname, "load_cache  PREP_TOTAL", t0, "from .prep/")
                return payload
        except Exception as e:
            _log(fname, f"cache load failed ({e}), re-computing", t0)

    try:
        import openmm.app as app
        import openmm as mm
        import openmm.unit as unit
        from pdbfixer import PDBFixer

        # ── Step 1: parse lines ───────────────────────────────────────────────
        raw_lines        = Path(pdb_in).read_text().splitlines()
        protein_lines    = []
        nonprotein_lines = []
        for line in raw_lines:
            rec = line[:6].strip()
            if rec in ("ATOM", "TER"):
                if rec == "ATOM" and len(line) > 20:
                    resname = line[17:20].strip()
                    # Skip UNK residues — no AMBER template
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
                # Skip CONECT — explicit bond records cause template mismatches
                # (e.g. disulfide bonds conflict with AMBER CYS/CYX templates)
                protein_lines.append(line)
        _log(fname, "parse", t0, f"{len(protein_lines)} protein lines")

        # ── Step 2: chain sort + renumber ─────────────────────────────────────
        chains_atoms: dict[str, list] = {}
        chain_order:  list[str]       = []
        for line in protein_lines:
            if line[:4] in ("ATOM", "HETA") and len(line) > 26:
                ch = line[21]
                if ch not in chains_atoms:
                    chains_atoms[ch] = []
                    chain_order.append(ch)
                chains_atoms[ch].append(line)

        renumbered: list[str] = []
        # _renumber_map[(chain_id, orig_resnum)] = new_resnum
        _renumber_map: dict[tuple[str, int], int] = {}
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
                    _renumber_map[(ch, orig_res)] = new_res_counter
                line = line[:22] + f"{new_res_counter:4d}" + line[26:]
                renumbered.append(line)
            renumbered.append("TER")
        protein_lines = renumbered
        _log(fname, "chain_sort", t0, f"{len(chain_order)} chains")

        # ── Step 3: PCA → GLN ─────────────────────────────────────────────────
        fixed_lines: list[str] = []
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

        # ── Step 4: PDBFixer ──────────────────────────────────────────────────
        fixer = PDBFixer(pdbfile=StringIO("\n".join(protein_lines)))
        _log(fname, "fixer_init", t0)

        fixer.findMissingResidues()
        _log(fname, "findMissingResidues", t0)

        # Don't fill large gaps (>5 residues) in antigen chains — creates unphysical geometry
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
        _log(fname, "replaceNonstandard", t0)

        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        _log(fname, "addMissingAtoms", t0)

        fixer.removeHeterogens(keepWater=False)
        _log(fname, "removeHeterogens", t0)

        fixer.addMissingHydrogens(PH)
        n_atoms_after_h = fixer.topology.getNumAtoms()
        _log(fname, "addMissingHydrogens", t0, f"{n_atoms_after_h} atoms")

        modeller = app.Modeller(fixer.topology, fixer.positions)

        # Strip terminal-atom artifacts added by PDBFixer to wrong residues:
        #   - H2/H3 on non-N-terminal residues  → causes NGLN template mismatch
        #   - SH (HG) on CYS involved in disulfide → causes CYS/CYX bond mismatch
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

        # ── Step 5: createSystem ──────────────────────────────────────────────
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
            # Re-apply H2/H3 stripping after rebuild
            bad_atoms2 = []
            for chain in modeller.topology.chains():
                residues2 = list(chain.residues())
                for i, res in enumerate(residues2):
                    for atom in res.atoms():
                        if atom.name in ("H2", "H3") and i != 0:
                            bad_atoms2.append(atom)
            if bad_atoms2:
                modeller.delete(bad_atoms2)
                _log(fname, f"stripped {len(bad_atoms2)} terminal-artifact atoms (retry)", t0)
            try:
                system = _create_system(modeller.topology)
                result["msg"] = f"terminal fix applied: {e1}"
                _log(fname, "createSystem (after fix)", t0, f"{system.getNumParticles()} particles")
            except Exception as e2:
                # Drop N-terminal GLN on each chain if template still fails
                _log(fname, "createSystem FAILED again — dropping N-term GLN", t0, str(e2)[:60])
                nterm_gln = [
                    a for ch in modeller.topology.chains()
                    for i, res in enumerate(list(ch.residues()))
                    for a in res.atoms()
                    if i == 0 and res.name == "GLN"
                ]
                if nterm_gln:
                    modeller.delete(nterm_gln)
                    _log(fname, f"dropped {len(nterm_gln)} N-term GLN atoms", t0)
                system = _create_system(modeller.topology)
                result["msg"] = f"terminal fix + drop N-term GLN: {e1}"
                _log(fname, "createSystem (after GLN drop)", t0, f"{system.getNumParticles()} particles")

        # ── Step 6: CA restraints ─────────────────────────────────────────────
        restraint = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter(
            "k", K_RESTRAINT_KJ * unit.kilojoules_per_mole / unit.nanometer**2
        )
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        # Load gap-filled residues (unrestrained) if gap_fills JSON exists
        _gap_json = Path(pdb_in).with_name(Path(pdb_in).stem + "_gap_fills.json")
        if _gap_json.exists():
            import json as _json
            _data = _json.loads(_gap_json.read_text())
            _unrestrained = set()
            for _fill in _data.get("fills", []):
                _ch = _fill["chain"]
                for _r in _fill.get("filled_residues", []):
                    _unrestrained.add((_ch, _r))
                _gs, _ge = _fill["gap_start"], _fill["gap_end"]
                for _buf in range(4):
                    _unrestrained.add((_ch, _gs - _buf))
                    _unrestrained.add((_ch, _ge + _buf))
        else:
            _unrestrained = set()

        pos_nm       = modeller.positions.value_in_unit(unit.nanometer)
        n_restrained = 0
        for atom in modeller.topology.atoms():
            if atom.name == "CA":
                _ch_id  = atom.residue.chain.id
                try:   _resseq = int(atom.residue.id)
                except: _resseq = -1
                if (_ch_id, _resseq) in _unrestrained:
                    continue   # gap-filled residue — no restraint
                x0, y0, z0 = pos_nm[atom.index]
                restraint.addParticle(atom.index, [x0, y0, z0])
                n_restrained += 1
        system.addForce(restraint)
        _log(fname, "CA_restraints", t0, f"{n_restrained} CA atoms")

        # ── Step 6b: strong anchor restraints (gap-split termini) ─────────────
        # Look for companion sidecar: <stem>_gap_split.json (same dir as input)
        _anchor_json = Path(pdb_in).with_name(Path(pdb_in).stem + "_gap_split.json")
        if _anchor_json.exists():
            import json as _json
            _adata = _json.loads(_anchor_json.read_text())
            # Build set of (chain_id, new_resnum) pairs — translate original
            # resnums through _renumber_map since Step 2 has already renumbered.
            _anchor_set: set[tuple[str, int]] = set()
            for _sp in _adata.get("splits", []):
                for (_ac, _ar) in [
                    (_sp["left_chain"],  _sp["left_anchor_resnum"]),
                    (_sp["right_chain"], _sp["right_anchor_resnum"]),
                ]:
                    _new_r = _renumber_map.get((_ac, _ar))
                    if _new_r is not None:
                        _anchor_set.add((_ac, _new_r))
                    else:
                        _log(fname, f"anchor ({_ac},{_ar}) not found in renumber map", t0)

            # Separate CustomExternalForce with hardcoded k so it survives
            # the clash-annealing loop that zeroes the global "k" parameter.
            K_ANCHOR_KJ = 5000.0  # kJ/mol/nm²
            anchor_force = mm.CustomExternalForce(
                f"{K_ANCHOR_KJ}*((x-ax0)^2+(y-ay0)^2+(z-az0)^2)"
            )
            anchor_force.addPerParticleParameter("ax0")
            anchor_force.addPerParticleParameter("ay0")
            anchor_force.addPerParticleParameter("az0")

            n_anchored = 0
            for atom in modeller.topology.atoms():
                if atom.name == "CA":
                    _ch_id = atom.residue.chain.id
                    try:   _resseq = int(atom.residue.id)
                    except: _resseq = -1
                    if (_ch_id, _resseq) in _anchor_set:
                        x0, y0, z0 = pos_nm[atom.index]
                        anchor_force.addParticle(atom.index, [x0, y0, z0])
                        n_anchored += 1
            system.addForce(anchor_force)
            _log(fname, "anchor_restraints", t0,
                 f"{n_anchored} anchor CA atoms  k={K_ANCHOR_KJ} kJ/mol/nm²")
        else:
            n_anchored = 0

        # ── Step 7: serialise ─────────────────────────────────────────────────
        system_xml   = mm.XmlSerializer.serialize(system)
        positions_nm = [[x, y, z] for x, y, z in pos_nm]

        orig_chains: list[str] = []
        for line in protein_lines:
            if line[:4] == "ATOM" and len(line) > 21 and line[21].strip():
                ch = line[21]
                if ch not in orig_chains:
                    orig_chains.append(ch)

        # Assign original chain IDs onto the topology BEFORE writing so
        # PDBFile.writeFile + keepIds preserves them. This avoids relying on
        # the post-hoc zip-based remap in _minimize_one, which can silently
        # mislabel chains when writeFile reassigns IDs alphabetically.
        _topo_chains = list(modeller.topology.chains())
        if len(_topo_chains) == len(orig_chains):
            for _tc, _orig_id in zip(_topo_chains, orig_chains):
                _tc.id = _orig_id

        topo_buf = StringIO()
        app.PDBFile.writeFile(modeller.topology, modeller.positions, topo_buf,
                              keepIds=True)
        topology_pdb = topo_buf.getvalue()

        xml_kb = len(system_xml) / 1024
        _log(fname, "serialise", t0, f"xml={xml_kb:.0f}KB  PREP_TOTAL")

        result.update({
            "status":           "READY",
            "system_xml":       system_xml,
            "positions_nm":     positions_nm,
            "topology_pdb":     topology_pdb,
            "orig_chains":      orig_chains,
            "nonprotein_lines": nonprotein_lines,
            "n_restrained":     n_restrained,
        })

        # ── Write to .prep cache ──────────────────────────────────────────────
        try:
            prep_dir.mkdir(parents=True, exist_ok=True)
            xml_path = prep_dir / (fname + ".xml")
            xml_path.write_text(system_xml)
            # Save pre-relaxed PDB (PDBFixer output before minimisation)
            prerelax_pdb_path = prep_dir / (fname + "_prerelax.pdb")
            prerelax_pdb_path.write_text(topology_pdb)
            cache_payload = {k: v for k, v in result.items() if k != "system_xml"}
            cache_payload["system_xml_path"] = str(xml_path)
            cache_payload["prerelax_pdb_path"] = str(prerelax_pdb_path)
            cache_json.write_text(_json.dumps(cache_payload))
            _log(fname, "cache_write", t0, f"→ .prep/")
        except Exception as ce:
            _log(fname, f"cache write failed ({ce})", t0)

    except Exception as e:
        result["msg"] = str(e)
        _log(fname, f"PREP-ERR: {e}", t0)
        traceback.print_exc()

    return result


# =============================================================================
# Stage 2 – GPU worker
# =============================================================================

def _minimize_one(prep: dict) -> dict:
    fname         = prep["file"]
    pdb_out       = prep["pdb_out"]
    platform_spec = prep["platform_spec"]
    result        = {"file": fname, "status": "ERROR", "msg": prep.get("msg", "")}
    t0 = time.time()

    try:
        import openmm.app as app
        import openmm as mm
        import openmm.unit as unit

        # ── Deserialise ───────────────────────────────────────────────────────
        # Load deferred system XML if CPU worker skipped it to save RAM
        if prep.get("system_xml") is None:
            xml_path = prep.get("system_xml_path", "")
            prep["system_xml"] = Path(xml_path).read_text()
        system = mm.XmlSerializer.deserialize(prep["system_xml"])
        _log(fname, "deserialise_xml", t0, f"{system.getNumParticles()} particles")

        positions = [mm.Vec3(x, y, z) * unit.nanometer for x, y, z in prep["positions_nm"]]

        topo_pdb = app.PDBFile(StringIO(prep["topology_pdb"]))
        topology = topo_pdb.topology
        _log(fname, "load_topology", t0)

        # ── Build simulation ──────────────────────────────────────────────────
        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picoseconds,
        )

        if platform_spec == "cpu":
            plat = mm.Platform.getPlatformByName("CPU")
            sim  = app.Simulation(topology, system, integrator, plat)
        else:
            try:
                plat = mm.Platform.getPlatformByName("CUDA")
                sim  = app.Simulation(topology, system, integrator, plat)
            except Exception as plat_err:
                _log(fname, f"CUDA failed ({plat_err}), using default", t0)
                sim  = app.Simulation(topology, system, integrator)

        actual_platform = sim.context.getPlatform().getName()
        _log(fname, "build_simulation", t0, f"platform={actual_platform}")

        # ── Minimise ──────────────────────────────────────────────────────────
        # Normal structures converge in ~100-200 iterations; 100 iterations is
        # enough to distinguish: clash structures barely move, normal ones drop.
        CLASH_PROBE_ITER = 100
        CLASH_THRESHOLD  = 1e6   # kJ/mol — if still this high after probe, restraints block resolution

        sim.context.setPositions(positions)
        e_before = (
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        _log(fname, "set_positions", t0, f"E_before={e_before:.1f} kJ/mol")

        # ── Pre-minimization: unrestrained clash resolution ──────────────
        # PDBFixer places hydrogens/atoms in idealized geometry that can
        # severely clash at chain interfaces. A short unrestrained pass
        # resolves these before the restrained L-BFGS, preventing NaN.
        PRE_MIN_THRESHOLD = 1e6  # only needed for high-energy structures
        if e_before > PRE_MIN_THRESHOLD:
            _log(fname, "high initial E — unrestrained pre-minimization", t0)
            sim.context.setParameter("k", 0.0)
            sim.minimizeEnergy(
                tolerance=10.0 * unit.kilojoules_per_mole,
                maxIterations=500,
            )
            e_pre = (
                sim.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
            _log(fname, "pre-min (unrestrained)", t0, f"E_pre={e_pre:.1f} kJ/mol")
            # Restore restraints
            sim.context.setParameter("k", K_RESTRAINT_KJ)

        # Probe pass with normal restraints
        sim.minimizeEnergy(
            tolerance=ENERGY_TOL_KJ * unit.kilojoules_per_mole,
            maxIterations=CLASH_PROBE_ITER,
        )
        e_probe = (
            sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoules_per_mole)
        )
        _log(fname, f"minimizeEnergy probe ({CLASH_PROBE_ITER} iter)", t0,
             f"E_probe={e_probe:.1f} kJ/mol")

        if e_probe > CLASH_THRESHOLD:
            # Restraints are preventing clash resolution — release gradually to
            # avoid coordinate divergence (NaN) on extreme clashes
            _log(fname, "clash unresolved — gradual restraint release", t0)
            for k_val in (K_RESTRAINT_KJ * 0.5,
                          K_RESTRAINT_KJ * 0.1,
                          K_RESTRAINT_KJ * 0.01,
                          0.0):
                sim.context.setParameter("k", k_val)
                sim.minimizeEnergy(
                    tolerance=ENERGY_TOL_KJ * unit.kilojoules_per_mole,
                    maxIterations=200,
                )
            e_free = (
                sim.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoules_per_mole)
            )
            _log(fname, "minimizeEnergy free (gradual)", t0, f"E_free={e_free:.1f} kJ/mol")
            # Restore restraints for final settling pass
            sim.context.setParameter("k", K_RESTRAINT_KJ)

        # Final pass (no-op if already converged; settles under restored restraints after clash fix)
        sim.minimizeEnergy(
            tolerance=ENERGY_TOL_KJ * unit.kilojoules_per_mole,
            maxIterations=MAX_ITER,
        )
        state   = sim.context.getState(getPositions=True, getEnergy=True)
        e_after = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        _log(fname, "minimizeEnergy", t0, f"E_after={e_after:.1f}  ΔE={e_after-e_before:+.0f} kJ/mol")

        if math.isnan(e_after):
            raise ValueError("NaN energy")

        minimized_pos = state.getPositions()

        # ── Write PDB ─────────────────────────────────────────────────────────
        # Defensively re-assign original chain IDs onto the topology in case
        # they were reset during PDBFile(StringIO(...)) reload. Combined with
        # keepIds=True this bypasses OpenMM's alphabetical-rename behavior.
        orig_chains = prep["orig_chains"]
        _topo_chains = list(topology.chains())
        if len(_topo_chains) == len(orig_chains):
            for _tc, _orig_id in zip(_topo_chains, orig_chains):
                _tc.id = _orig_id

        protein_buf = StringIO()
        app.PDBFile.writeFile(topology, minimized_pos, protein_buf, keepIds=True)
        protein_pdb = protein_buf.getvalue()

        # Verify chain topology matches expectation
        omm_chains: list[str] = []
        for line in protein_pdb.splitlines():
            if line[:4] == "ATOM" and len(line) > 21:
                ch = line[21]
                if ch not in omm_chains:
                    omm_chains.append(ch)

        if omm_chains != orig_chains:
            # Fall back to legacy zip remap (positional mapping) if the
            # explicit assignment somehow failed, but only when counts match.
            if omm_chains and len(omm_chains) == len(orig_chains):
                ch_map = dict(zip(omm_chains, orig_chains))
                remapped = []
                for line in protein_pdb.splitlines():
                    if line[:4] in ("ATOM", "HETA", "TER ") and len(line) > 21:
                        line = line[:21] + ch_map.get(line[21], line[21]) + line[22:]
                    remapped.append(line)
                protein_pdb = "\n".join(remapped)
            else:
                raise ValueError(
                    f"chain topology mismatch after relax: omm={omm_chains} "
                    f"orig={orig_chains}"
                )

        n_restrained = prep["n_restrained"]
        pdb_out_path = Path(pdb_out)
        pdb_out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pdb_out_path, "w") as fh:
            fh.write(
                f"REMARK  OpenMM AMBER14+OBC2 minimization  "
                f"E_before={e_before:.1f} E_after={e_after:.1f} kJ/mol\n"
            )
            fh.write(
                f"REMARK  CA restraints k={K_RESTRAINT_KJ} kJ/mol/nm2  "
                f"n_restrained={n_restrained}  max_iter={MAX_ITER}\n"
            )
            for line in protein_pdb.splitlines():
                if line.startswith(("ATOM", "TER")):
                    fh.write(line + "\n")
            fh.write("END\n")

        _log(fname, "write_pdb  GPU_TOTAL", t0)

        result.update({
            "status":       "OK",
            "e_before":     e_before,
            "e_after":      e_after,
            "n_restrained": n_restrained,
        })

    except Exception as e:
        result["msg"] = str(e)
        _log(fname, f"GPU-ERR: {e}", t0)
        traceback.print_exc()

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",      type=Path, required=True)
    parser.add_argument("--output",     type=Path, required=True)
    parser.add_argument("--nproc-cpu",  type=int,  default=6)
    parser.add_argument("--nproc-gpu",  type=int,  default=2)
    parser.add_argument("--platform",   choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--prep-only",  action="store_true",
                        help="Run CPU prep + cache only; skip GPU minimization")
    parser.add_argument("--files",      type=Path, default=None,
                        help="Text file listing PDB filenames to process (one per "
                             "line). If absent, glob all *.pdb in --input.")
    args = parser.parse_args()

    ckpt_file = args.output / ".relax_ckpt.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if ckpt_file.exists():
        for line in ckpt_file.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("status") == "OK":
                    done.add(r["file"])
            except Exception:
                pass
    print(f"Already done : {len(done)}  platform={args.platform}", flush=True)

    # Select candidate PDBs: explicit --files list if given, else glob input dir
    if args.files is not None and args.files.exists():
        _names = [ln.strip() for ln in args.files.read_text().splitlines() if ln.strip()]
        _pdbs = [args.input / n for n in _names if (args.input / n).exists()]
        _missing = [n for n in _names if not (args.input / n).exists()]
        if _missing:
            print(f"WARNING: {len(_missing)} files in --files not found in {args.input}:",
                  flush=True)
            for n in _missing[:10]:
                print(f"  MISSING: {n}", flush=True)
        _pdbs = sorted(_pdbs)
    else:
        _pdbs = sorted(args.input.glob("*.pdb"))

    jobs = [
        (str(pdb), str(args.output / pdb.name), args.platform)
        for pdb in _pdbs
        if pdb.name not in done
    ]
    print(f"Jobs to run  : {len(jobs)}  (cpu={args.nproc_cpu}  gpu={args.nproc_gpu})", flush=True)

    if not jobs:
        print("Nothing to do!", flush=True)
        return

    # Tee stdout+stderr to log file BEFORE forking pools so workers inherit it
    log_path = args.output / "relax.log"
    _log_fh  = open(log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
    print(f"=== relax started  log={log_path} ===", flush=True)

    n_ok = n_err = 0
    ckpt_lock = threading.Lock()

    def _write_ckpt(r):
        nonlocal n_ok, n_err
        with ckpt_lock:
            with open(ckpt_file, "a") as ckpt:
                json.dump(r, ckpt)
                ckpt.write("\n")
            if r["status"] == "OK":
                n_ok += 1
            else:
                n_err += 1
            total = n_ok + n_err
            if total % 50 == 0:
                print(f"  Progress: {total}/{len(jobs)} — {n_ok} OK, {n_err} ERR", flush=True)

    if args.prep_only:
        # ── Prep-only pass: fill .prep/ cache using all CPU workers ──────────
        print(f"Mode: PREP-ONLY  (nproc-cpu={args.nproc_cpu})", flush=True)
        n_prep_ok = n_prep_err = 0
        cpu_pool = mp.Pool(args.nproc_cpu)
        try:
            for i, prep in enumerate(
                    cpu_pool.imap_unordered(_prepare_one, jobs, chunksize=1), 1):
                if prep["status"] == "ERROR":
                    n_prep_err += 1
                    print(f"  PREP-ERR [{i}/{len(jobs)}] {prep['file']}: {prep['msg']}", flush=True)
                else:
                    n_prep_ok += 1
                    if i % 100 == 0 or i == len(jobs):
                        print(f"  [{i}/{len(jobs)}] prep_ok={n_prep_ok} err={n_prep_err}", flush=True)
        finally:
            cpu_pool.terminate()
            cpu_pool.join()
        print(f"\nPrep done: {n_prep_ok} cached, {n_prep_err} errors → {args.output}/.prep/", flush=True)
        sys.exit(0)

    # ── Normal (or cache-assisted) GPU minimization pass ─────────────────────
    # When .prep/ cache exists for a file, CPU worker returns instantly from
    # cache; GPU queue drains quickly without CPU bottleneck.
    # Semaphore prevents queuing more GPU jobs than this limit at once.
    # Each prep dict holds positions_nm + topology_pdb (~3-5 MB); without a
    # limit the main loop dispatches all 2000+ jobs immediately, blowing RAM.
    # When fully cached, raise the limit so GPU workers stay saturated.
    prep_dir = args.output / ".prep"
    n_cached = sum(1 for j in jobs if (prep_dir / (Path(j[0]).name + ".json")).exists())
    fully_cached = (n_cached == len(jobs))
    GPU_QUEUE_LIMIT = len(jobs) if fully_cached else args.nproc_gpu * 4
    print(f"Cached preps: {n_cached}/{len(jobs)}  gpu_queue_limit={GPU_QUEUE_LIMIT}", flush=True)

    gpu_sem = threading.Semaphore(GPU_QUEUE_LIMIT)

    def _gpu_done(r):
        gpu_sem.release()
        _write_ckpt(r)

    cpu_pool = mp.Pool(args.nproc_cpu)
    gpu_pool = mp.Pool(args.nproc_gpu)
    try:
        gpu_async_results = []

        for prep in cpu_pool.imap_unordered(_prepare_one, jobs, chunksize=1):
            if prep["status"] == "ERROR":
                _write_ckpt({"file": prep["file"], "status": "ERROR", "msg": prep["msg"]})
            else:
                gpu_sem.acquire()   # blocks when GPU queue is full
                ar = gpu_pool.apply_async(_minimize_one, (prep,), callback=_gpu_done)
                gpu_async_results.append(ar)

        for ar in gpu_async_results:
            ar.wait()
    finally:
        cpu_pool.terminate()
        cpu_pool.join()
        gpu_pool.terminate()
        gpu_pool.join()

    print(f"\nDone: {n_ok} OK, {n_err} errors → {args.output}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
