#!/usr/bin/env python3
"""sc_connolly.py — Shape complementarity (Lawrence & Colman 1993) via
grid-based Connolly Solvent Excluded Surface (SES).

Replaces the prior SAS-based implementation that achieved only Pearson r~0.50
vs Rosetta/MSMS.  The key improvement: the SES includes concave re-entrant
saddle patches (where the probe rolls between two atoms).  These saddle normals
point sideways/away from the protein and correctly penalise low-complementarity
interfaces.  Without them, SC values are systematically inflated.

Algorithm
---------
1. Parse the PDB; extract heavy atoms for each chain.
2. Build a signed-distance field on a regular grid:
   - Voxel grid with spacing `grid_spacing` Å.
   - VDW occupancy mask: voxel inside any atom sphere ← True.
   - EDT inside protein + EDT outside protein → signed distance field.
3. SES = isosurface of (signed_dist - probe_radius) == 0, extracted via
   marching cubes (skimage.measure).  Normals from the gradient of the SES
   field — these include the re-entrant saddle patches.
4. Interface filter: keep SES vertices within `interface_cutoff_prefilter` Å
   of any heavy atom of the opposing chain.
5. SC(A→B): for each interface vertex on A, find nearest interface vertex on B
   (KDTree), accept if distance < `interface_cutoff` Å, accumulate
   dot(n_A, -n_B).  Symmetric B→A.
6. SC = mean(median of positive dot products A→B, median B→A).
   Clipped to [0, 1].

References
----------
Connolly ML (1983) Solvent-accessible surfaces of proteins and nucleic acids.
  Science 221:709-713. doi:10.1126/science.6879170
Lawrence MC, Colman PM (1993) Shape complementarity at protein/protein
  interfaces. J Mol Biol 234:946-950. doi:10.1006/jmbi.1993.1648

Dependencies (all pip-installable)
-----------------------------------
  numpy, scipy, scikit-image, trimesh, biopython, scikit-learn

Usage
-----
As a library:
    from sc_connolly import compute_sc
    sc = compute_sc("complex.pdb", chain_vhh="H", chain_ag="A")

CLI (single):
    python sc_connolly.py --pdb complex.pdb --vhh-chain H --ag-chain A

CLI (batch):
    python sc_connolly.py --batch-dir data/positives/ \
        --output sc_results.csv --workers 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import marching_cubes
from sklearn.neighbors import KDTree

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Van der Waals radii (Å) keyed by element string (upper-case)
# Source: Bondi (1964) with common additions; matches Rosetta defaults.
# ---------------------------------------------------------------------------
_VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
    "P": 1.80, "H": 1.20, "F": 1.47, "CL": 1.75,
    "BR": 1.85, "I":  1.98, "SE": 1.90, "FE": 1.40,
    "ZN": 1.39, "MG": 1.73, "CA": 1.74, "NA": 2.27,
    "K":  2.75, "MN": 1.61, "CU": 1.40, "NI": 1.63,
    "CO": 1.40, "CD": 1.58, "HG": 1.55, "default": 1.70,
}


# ---------------------------------------------------------------------------
# SES construction via grid distance transform + marching cubes
# ---------------------------------------------------------------------------

def build_ses_mesh(
    atom_coords: np.ndarray,       # (N, 3) heavy-atom Cartesian coordinates
    atom_radii: np.ndarray,        # (N,)   VDW radii
    probe_radius: float = 1.4,
    grid_spacing: float = 0.5,
    padding: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the Connolly SES as a triangulated mesh.

    Returns
    -------
    verts   : (M, 3) vertex coordinates in Å
    faces   : (F, 3) triangle vertex indices
    normals : (M, 3) outward unit normals at each vertex (toward solvent)
    """
    lo = atom_coords.min(axis=0) - padding
    hi = atom_coords.max(axis=0) + padding
    shape = tuple((np.ceil((hi - lo) / grid_spacing).astype(int) + 1).tolist())

    # ---- Build VDW occupancy mask (vectorised via KDTree) -------------------
    # Create grid point array for all voxels
    nx, ny, nz = shape
    # Use mgrid to get voxel centres
    xi = np.arange(nx) * grid_spacing + lo[0]
    yi = np.arange(ny) * grid_spacing + lo[1]
    zi = np.arange(nz) * grid_spacing + lo[2]

    # KDTree on atoms; for each voxel find nearest atom and check if inside
    tree_atoms = KDTree(atom_coords)

    # Process in chunks to avoid huge memory allocation
    chunk = 500_000
    n_vox = nx * ny * nz
    protein_mask = np.zeros(n_vox, dtype=bool)

    # Build grid pts on the fly in chunks
    # Generate flat index → 3D index mapping
    grid_pts_flat = np.empty((chunk, 3), dtype=np.float32)

    # We use the mgrid approach but chunked
    idx_all = np.arange(n_vox)
    for start in range(0, n_vox, chunk):
        end = min(start + chunk, n_vox)
        idx_chunk = idx_all[start:end]
        ix = idx_chunk // (ny * nz)
        iy = (idx_chunk % (ny * nz)) // nz
        iz = idx_chunk % nz
        pts = np.stack([xi[ix], yi[iy], zi[iz]], axis=1)
        # Query nearest atom
        dists, idx_near = tree_atoms.query(pts, k=1)
        dists = dists[:, 0]
        idx_near = idx_near[:, 0]
        # Inside VDW sphere of nearest atom
        inside = dists <= atom_radii[idx_near]
        protein_mask[start:end] = inside

    protein_mask = protein_mask.reshape(shape)

    # ---- Signed distance field ----------------------------------------------
    # EDT inside protein (distance from protein-interior voxels to protein surface)
    edt_inside = distance_transform_edt(protein_mask) * grid_spacing
    # EDT outside protein (distance from exterior voxels to protein surface)
    edt_outside = distance_transform_edt(~protein_mask) * grid_spacing
    # signed_dist: positive outside protein, negative inside
    signed_dist = edt_outside - edt_inside

    # ---- SES field: isosurface at probe_radius from protein surface ---------
    # The SES is the boundary of the region inaccessible to probe centres.
    # Probe centre accessible iff signed_dist >= probe_radius.
    # SES = isosurface of (signed_dist - probe_radius) == 0
    ses_field = signed_dist - probe_radius

    # ---- Marching cubes -----------------------------------------------------
    # Level=0 extracts the SES; normals point toward increasing ses_field
    # = toward solvent (outward) since ses_field increases away from protein.
    try:
        verts, faces, normals, _ = marching_cubes(
            ses_field, level=0.0,
            spacing=(grid_spacing, grid_spacing, grid_spacing),
            allow_degenerate=False,
        )
    except Exception as exc:
        logger.warning("marching_cubes failed: %s", exc)
        return np.empty((0, 3)), np.empty((0, 3), dtype=int), np.empty((0, 3))

    # Shift vertices back to real-world coordinates
    verts = verts + lo

    # Normalise normals
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    lens = np.where(lens < 1e-12, 1.0, lens)
    normals = normals / lens

    return verts, faces, normals


# ---------------------------------------------------------------------------
# SC computation from two SES meshes
# ---------------------------------------------------------------------------

def compute_sc_from_mesh(
    verts_A: np.ndarray,
    normals_A: np.ndarray,
    verts_B: np.ndarray,
    normals_B: np.ndarray,
    interface_cutoff: float = 3.5,
) -> float:
    """
    Compute SC from two SES meshes.

    Uses KDTree nearest-neighbour lookup (no ray casting required).

    Parameters
    ----------
    verts_A, verts_B : (M, 3) SES vertex coordinates
    normals_A, normals_B : (M, 3) outward unit normals
    interface_cutoff : Å — maximum surface-to-surface distance to include

    Returns
    -------
    float in [0, 1] or NaN
    """
    if len(verts_A) < 10 or len(verts_B) < 10:
        return float("nan")

    tree_B = KDTree(verts_B)
    dists_AB, idx_AB = tree_B.query(verts_A, k=1)
    mask_AB = dists_AB[:, 0] < interface_cutoff

    tree_A = KDTree(verts_A)
    dists_BA, idx_BA = tree_A.query(verts_B, k=1)
    mask_BA = dists_BA[:, 0] < interface_cutoff

    if mask_AB.sum() < 10 or mask_BA.sum() < 10:
        return float("nan")

    # Dot product: normal on A · (−normal on nearest B) → 1 = perfect complement
    dots_AB = np.einsum("ij,ij->i",
                         normals_A[mask_AB],
                         -normals_B[idx_AB[mask_AB, 0]])

    dots_BA = np.einsum("ij,ij->i",
                         normals_B[mask_BA],
                         -normals_A[idx_BA[mask_BA, 0]])

    # Lawrence & Colman: median of positive dot products
    pos_AB = dots_AB[dots_AB > 0]
    pos_BA = dots_BA[dots_BA > 0]

    sc_AB = float(np.median(pos_AB)) if len(pos_AB) > 0 else float("nan")
    sc_BA = float(np.median(pos_BA)) if len(pos_BA) > 0 else float("nan")

    vals = [v for v in [sc_AB, sc_BA] if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.clip(np.mean(vals), 0.0, 1.0))


# ---------------------------------------------------------------------------
# PDB parsing utilities
# ---------------------------------------------------------------------------

def _extract_chain_atoms(
    model, chain_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords, radii) for heavy atoms in chain_id."""
    if chain_id not in model:
        return np.empty((0, 3)), np.empty((0,))
    chain = model[chain_id]
    coords_list, radii_list = [], []
    for atom in chain.get_atoms():
        if atom.element == "H":
            continue
        elem = (atom.element or "C").strip().upper()
        r = _VDW.get(elem, _VDW["default"])
        coords_list.append(atom.coord)
        radii_list.append(r)
    if not coords_list:
        return np.empty((0, 3)), np.empty((0,))
    return np.array(coords_list, dtype=float), np.array(radii_list, dtype=float)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _extract_all_antigen_atoms(
    model, chain_vhh: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords, radii) for all heavy atoms in every non-VHH chain combined.

    Mirrors Rosetta's jump-based partitioning: the antigen side is the union of
    all chains that are not the VHH, so multi-chain antigens are handled correctly.
    """
    coords_list, radii_list = [], []
    for chain in model:
        if chain.id == chain_vhh:
            continue
        c, r = _extract_chain_atoms(model, chain.id)
        if len(c) > 0:
            coords_list.append(c)
            radii_list.append(r)
    if not coords_list:
        return np.empty((0, 3)), np.empty((0,))
    return np.concatenate(coords_list), np.concatenate(radii_list)


def compute_sc(
    pdb_path: str,
    chain_vhh: str = "H",
    chain_ag: str = "all",
    probe_radius: float = 1.4,
    grid_spacing: float = 0.5,
    padding: float = 5.0,
    interface_cutoff: float = 3.5,
    interface_cutoff_prefilter: float = 2.5,
    return_timing: bool = False,
) -> float:
    """
    Compute shape complementarity (Lawrence & Colman 1993) via grid-based SES.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file (VHH–antigen complex).
    chain_vhh : str
        Chain ID of the VHH nanobody (default "H").
    chain_ag : str
        Chain ID of the antigen, "all" to use all non-VHH chains combined
        (Rosetta-style jump partitioning, default), or "auto" for closest chain.
    probe_radius : float
        Solvent probe radius in Å (default 1.4, water).
    grid_spacing : float
        Voxel size in Å for SES construction (default 0.5).
    padding : float
        Grid padding beyond atom extent in Å (default 5.0).
    interface_cutoff : float
        Surface-to-surface distance cutoff for SC scoring (default 3.5 Å).
    interface_cutoff_prefilter : float
        Pre-filter: keep SES vertices within this distance of opposing
        chain heavy atoms (default 2.5 Å).
    return_timing : bool
        If True, print timing to stderr.

    Returns
    -------
    float
        Shape complementarity in [0, 1], or NaN on failure.
    """
    from Bio import PDB

    t0 = time.time()

    # ---- 1. Parse PDB -------------------------------------------------------
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception as exc:
        logger.warning("PDB parse error for %s: %s", pdb_path, exc)
        return float("nan")

    try:
        model = next(structure.get_models())
    except StopIteration:
        logger.warning("No models in %s", pdb_path)
        return float("nan")

    coords_A, radii_A = _extract_chain_atoms(model, chain_vhh)

    # Antigen: all non-VHH chains combined (Rosetta jump-partitioning approach)
    if chain_ag == "all":
        coords_B, radii_B = _extract_all_antigen_atoms(model, chain_vhh)
    elif chain_ag == "auto":
        # Legacy: pick closest single chain
        best, best_d = None, float("inf")
        for ch in model:
            if ch.id == chain_vhh:
                continue
            c, _ = _extract_chain_atoms(model, ch.id)
            if len(c) == 0:
                continue
            d = float(KDTree(c).query(coords_A, k=1)[0].min())
            if d < best_d:
                best_d, best = d, ch.id
        if best is None:
            logger.warning("Cannot find antigen chain in %s", pdb_path)
            return float("nan")
        coords_B, radii_B = _extract_chain_atoms(model, best)
    else:
        coords_B, radii_B = _extract_chain_atoms(model, chain_ag)

    if len(coords_A) < 5 or len(coords_B) < 5:
        logger.warning("Too few atoms in %s for chain %s or antigen",
                       pdb_path, chain_vhh)
        return float("nan")

    # ---- 2. Build per-chain SES meshes --------------------------------------
    verts_A, faces_A, normals_A = build_ses_mesh(
        coords_A, radii_A,
        probe_radius=probe_radius,
        grid_spacing=grid_spacing,
        padding=padding,
    )

    verts_B, faces_B, normals_B = build_ses_mesh(
        coords_B, radii_B,
        probe_radius=probe_radius,
        grid_spacing=grid_spacing,
        padding=padding,
    )

    if len(verts_A) < 10 or len(verts_B) < 10:
        logger.warning("Empty SES for %s", pdb_path)
        return float("nan")

    # ---- 3. Interface pre-filter: keep SES vertices near opposing chain -----
    tree_B_atoms = KDTree(coords_B)
    dist_A2B, _ = tree_B_atoms.query(verts_A, k=1)
    iface_mask_A = dist_A2B[:, 0] < interface_cutoff_prefilter
    iface_verts_A = verts_A[iface_mask_A]
    iface_nrms_A = normals_A[iface_mask_A]

    tree_A_atoms = KDTree(coords_A)
    dist_B2A, _ = tree_A_atoms.query(verts_B, k=1)
    iface_mask_B = dist_B2A[:, 0] < interface_cutoff_prefilter
    iface_verts_B = verts_B[iface_mask_B]
    iface_nrms_B = normals_B[iface_mask_B]

    # Fallback: widen prefilter progressively for near-absent interfaces
    _fallback_prefilters = [3.5, 5.0, 7.0, 8.0]
    _pf = interface_cutoff_prefilter
    while (len(iface_verts_A) < 10 or len(iface_verts_B) < 10) and _fallback_prefilters:
        _pf = _fallback_prefilters.pop(0)
        logger.warning(
            "Degenerate interface in %s (%d/%d vertices at %.1fA) — retrying at %.1fA",
            pdb_path, len(iface_verts_A), len(iface_verts_B),
            interface_cutoff_prefilter, _pf)
        iface_mask_A = dist_A2B[:, 0] < _pf
        iface_verts_A = verts_A[iface_mask_A]
        iface_nrms_A = normals_A[iface_mask_A]
        iface_mask_B = dist_B2A[:, 0] < _pf
        iface_verts_B = verts_B[iface_mask_B]
        iface_nrms_B = normals_B[iface_mask_B]

    if len(iface_verts_A) < 10 or len(iface_verts_B) < 10:
        # Last-resort recovery: if chain_ag='all' combined all antigen chains and
        # none are near the VHH (e.g., wrong chain included or coordinate mismatch),
        # try each antigen chain individually and use the closest one.
        if chain_ag == "all":
            logger.warning(
                "Degenerate interface in %s at 8.0 Å — trying per-chain closest-partner "
                "recovery (possible wrong antigen chain in PDB)",
                pdb_path)
            best_min_d = float("inf")
            best_chain_id: Optional[str] = None
            best_coords_B: Optional[np.ndarray] = None
            best_radii_B: Optional[np.ndarray] = None
            for ch in model:
                if ch.id == chain_vhh:
                    continue
                c_ch, r_ch = _extract_chain_atoms(model, ch.id)
                if len(c_ch) < 5:
                    continue
                tree_ch = KDTree(c_ch)
                d_min = float(tree_ch.query(coords_A, k=1)[0].min())
                if d_min < best_min_d:
                    best_min_d = d_min
                    best_chain_id = ch.id
                    best_coords_B = c_ch
                    best_radii_B = r_ch
            if best_min_d < float("inf"):
                logger.warning(
                    "Closest antigen chain to VHH is '%s' (min dist %.2f Å) — "
                    "rebuilding SES for closest-partner recovery",
                    best_chain_id, best_min_d)
                verts_B, _, normals_B = build_ses_mesh(
                    best_coords_B, best_radii_B,
                    probe_radius=probe_radius,
                    grid_spacing=grid_spacing,
                    padding=padding,
                )
                tree_B_atoms = KDTree(best_coords_B)
                dist_A2B, _ = tree_B_atoms.query(verts_A, k=1)
                dist_B2A, _ = tree_A_atoms.query(verts_B, k=1)
                for _pf in [2.5, 3.5, 5.0, 7.0, 8.0]:
                    iface_verts_A = verts_A[dist_A2B[:, 0] < _pf]
                    iface_nrms_A = normals_A[dist_A2B[:, 0] < _pf]
                    iface_verts_B = verts_B[dist_B2A[:, 0] < _pf]
                    iface_nrms_B = normals_B[dist_B2A[:, 0] < _pf]
                    if len(iface_verts_A) >= 10 and len(iface_verts_B) >= 10:
                        break
        if len(iface_verts_A) < 10 or len(iface_verts_B) < 10:
            logger.warning(
                "Degenerate interface in %s: %d/%d interface SES vertices — returning NaN",
                pdb_path, len(iface_verts_A), len(iface_verts_B))
            return float("nan")

    # ---- 4. Compute SC -------------------------------------------------------
    sc = compute_sc_from_mesh(
        iface_verts_A, iface_nrms_A,
        iface_verts_B, iface_nrms_B,
        interface_cutoff=interface_cutoff,
    )

    elapsed = time.time() - t0
    if return_timing:
        print(f"  [timing] {Path(pdb_path).name}: {elapsed:.1f}s", file=sys.stderr)

    return sc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _worker(args_tuple):
    pdb_path, chain_vhh, chain_ag, kwargs = args_tuple
    try:
        sc = compute_sc(pdb_path, chain_vhh=chain_vhh, chain_ag=chain_ag, **kwargs)
    except Exception as exc:
        sc = float("nan")
        logger.error("Error processing %s: %s", pdb_path, exc)
    stem = Path(pdb_path).name
    return stem, sc


def _cli_single(args: argparse.Namespace) -> None:
    t0 = time.time()
    sc = compute_sc(
        pdb_path=args.pdb,
        chain_vhh=args.vhh_chain,
        chain_ag=args.ag_chain,
        probe_radius=args.probe_radius,
        grid_spacing=args.grid_spacing,
        interface_cutoff=args.interface_cutoff,
        interface_cutoff_prefilter=args.interface_cutoff_prefilter,
    )
    elapsed = time.time() - t0
    print(f"SC = {sc:.6f}  ({elapsed:.1f}s)")


def _cli_batch(args: argparse.Namespace) -> None:
    batch_dir = Path(args.batch_dir)
    pdbs = sorted(batch_dir.glob("*.pdb"))
    if not pdbs:
        print(f"No PDB files found in {batch_dir}", file=sys.stderr)
        sys.exit(1)

    kwargs = dict(
        probe_radius=args.probe_radius,
        grid_spacing=args.grid_spacing,
        interface_cutoff=args.interface_cutoff,
        interface_cutoff_prefilter=args.interface_cutoff_prefilter,
    )

    tasks = [(str(p), args.vhh_chain, args.ag_chain, kwargs) for p in pdbs]

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, t): t[0] for t in tasks}
        try:
            from tqdm import tqdm
            for fut in tqdm(as_completed(futures), total=len(futures)):
                results.append(fut.result())
        except ImportError:
            for fut in as_completed(futures):
                results.append(fut.result())

    results.sort(key=lambda x: x[0])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "sc_ses"])
        writer.writerows(results)
    print(f"Wrote {len(results)} results to {out_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    # Detect mode
    if "--batch-dir" in sys.argv:
        parser = argparse.ArgumentParser(
            description="SC batch mode (SES-based)")
        parser.add_argument("--batch-dir", required=True)
        parser.add_argument("--output", required=True)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--vhh-chain", default="H", dest="vhh_chain")
        parser.add_argument("--ag-chain", default="A", dest="ag_chain")
        parser.add_argument("--probe-radius", type=float, default=1.4,
                            dest="probe_radius")
        parser.add_argument("--grid-spacing", type=float, default=0.5,
                            dest="grid_spacing")
        parser.add_argument("--interface-cutoff", type=float, default=3.5,
                            dest="interface_cutoff")
        parser.add_argument("--interface-cutoff-prefilter", type=float, default=2.5,
                            dest="interface_cutoff_prefilter")
        args = parser.parse_args()
        _cli_batch(args)
    else:
        parser = argparse.ArgumentParser(
            description="SC single-structure mode (SES-based)")
        parser.add_argument("--pdb", required=True)
        parser.add_argument("--vhh-chain", default="H", dest="vhh_chain")
        parser.add_argument("--ag-chain", default="A", dest="ag_chain")
        parser.add_argument("--probe-radius", type=float, default=1.4,
                            dest="probe_radius")
        parser.add_argument("--grid-spacing", type=float, default=0.5,
                            dest="grid_spacing")
        parser.add_argument("--interface-cutoff", type=float, default=3.5,
                            dest="interface_cutoff")
        parser.add_argument("--interface-cutoff-prefilter", type=float, default=2.5,
                            dest="interface_cutoff_prefilter")
        args = parser.parse_args()
        _cli_single(args)


if __name__ == "__main__":
    main()
