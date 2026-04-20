#!/usr/bin/env python3
"""score.py — OpenBinder high-throughput inference entry point.

Takes a directory of VHH–antigen PDB files (e.g. outputs from RFAntibody,
BoltzGen, or IgGM), runs the full pipeline, and outputs binding-probability
scores plus all intermediate files in organized subdirectories.

Pipeline steps
--------------
1. Validate input: scan --input-dir for *.pdb
2. Restrained relaxation  → intermediates/relaxed_rest/   (OpenMM AMBER14+OBC2)
3. Unrestrained relaxation → intermediates/relaxed_unrest/
4. OpenMM feature extraction → intermediates/features_openmm.csv
5. COCaDA feature extraction → intermediates/features_cocada.csv
6. ESM-2 feature extraction  → intermediates/features_esm.csv
7. Feature assembly via feature_combine.assemble_matrix (both_all, 153 dims)
8. Load checkpoint(s) and score
9. Write scores.csv sorted by ensemble_score descending

Usage
-----
    python scripts/v3/score.py \\
        --input-dir /path/to/pdbs \\
        --output-dir /path/to/results \\
        --mode both \\
        --device auto \\
        --workers 4 \\
        --checkpoint-dir models/checkpoints

    # If PDBs are already relaxed (rest+unrest subdirs populated):
    python scripts/v3/score.py \\
        --input-dir /path/to/pdbs \\
        --output-dir /path/to/results \\
        --skip-relaxation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from io import StringIO
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolve script dir and project root ─────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent    # Open-Binder/

# Add scripts/v3/ to path so feature_combine / datamodule import cleanly
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ── Constants ────────────────────────────────────────────────────────────────
# ESM-PPI column names produced by 07_extract_esm_ppi.py
ESM_VHH_DIM = 1280
ESM_AG_DIM  = 1280
ESM_CA_DIM  = 660
ESM_TOTAL_DIM = ESM_VHH_DIM + ESM_AG_DIM + ESM_CA_DIM   # 3220

# ESM-PPI checkpoint (esm_ppi_650m_ab.pth) — downloaded from Zenodo.
# Resolved via (in order): --esm-checkpoint CLI flag → $ESM_CHECKPOINT env var.
# Run `python scripts/download_assets.py --esm` to fetch the checkpoint automatically.
# Set COCADA_ROOT env var to the local COCaDA checkout, or pass --cocada-path.
ESM_CKPT    = os.environ.get("ESM_CHECKPOINT", "")
COCADA_PATH = os.environ.get("COCADA_ROOT", "")

# amino-acid three-to-one map (for ESM sequence extraction)
AA3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-dir", type=Path, required=True,
        help="Directory containing VHH–antigen PDB files (*.pdb).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root output directory (created if absent).",
    )
    p.add_argument(
        "--mode", choices=["rf", "mlp", "both"], default="both",
        help="Which model(s) to use (default: both).",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu", "auto"], default="auto",
        help="Device for MLP (default: auto → CUDA if available else CPU). "
             "RF always runs on CPU.",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Parallel workers for relaxation (default: cpu_count()).",
    )
    p.add_argument(
        "--skip-relaxation", action="store_true",
        help="Skip relaxation if intermediates/relaxed_rest/ and "
             "intermediates/relaxed_unrest/ are already populated.",
    )
    p.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory containing model checkpoint subdirs "
             "(default: models/checkpoints/ relative to script root).",
    )
    p.add_argument(
        "--vhh-chain", default="H",
        help="Chain ID of the VHH in the input PDB files (default: H).",
    )
    p.add_argument(
        "--training-data-dir", type=Path, default=None,
        help="Directory containing training feature CSVs used to fit the ESM PCA "
             "projection. Default: data/ relative to script root. Required unless "
             "--skip-relaxation and pre-projected ESM features already exist.",
    )
    p.add_argument(
        "--esm-checkpoint", type=Path, default=None,
        help="Path to the ESM-PPI checkpoint file (esm_ppi_650m_ab.pth). "
             "Overrides the $ESM_CHECKPOINT env var; or use "
             "`python scripts/download_assets.py --esm` to download. "
             "Download from: https://zenodo.org/records/16909543",
    )
    p.add_argument(
        "--cocada-path", type=Path, default=None,
        help="Path to the COCaDA repository root (must contain cocada.py). "
             "Overrides the COCADA_ROOT environment variable. "
             "Clone from: https://github.com/rplemos/COCaDA",
    )
    return p.parse_args()


# =============================================================================
# Logging setup
# =============================================================================

def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("openbinder.score")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# =============================================================================
# Step 2 & 3 — Relaxation (thin wrappers calling 02_relax.py internals)
# =============================================================================

def _relax_restrained_worker(args: tuple) -> dict:
    """Worker for ProcessPoolExecutor: restrained relaxation of one PDB."""
    pdb_in, pdb_out, platform = args
    fname = Path(pdb_in).name
    # Skip if already done
    if Path(pdb_out).exists():
        return {"file": fname, "status": "SKIPPED"}

    # Inline the prepare+minimize logic from 02_relax.py
    # We import inside the worker so the heavy OpenMM import only happens
    # in the subprocess, not in the main process.
    try:
        src_dir = str(Path(__file__).resolve().parent.parent.parent /
                      "scripts" / "relaxation")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Import the worker functions from the source script
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "relax_mod",
            Path(src_dir) / "02_relax.py",
        )
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        Path(pdb_out).parent.mkdir(parents=True, exist_ok=True)
        prep = mod._prepare_one((pdb_in, pdb_out, platform))
        if prep["status"] != "READY":
            return {"file": fname, "status": "ERROR", "msg": prep.get("msg", "prep failed")}
        result = mod._minimize_one(prep)
        return {"file": fname, "status": result["status"],
                "msg": result.get("msg", "")}
    except Exception as e:
        return {"file": fname, "status": "ERROR", "msg": str(e)}


def _relax_unrestrained_worker(args: tuple) -> dict:
    """Worker for ProcessPoolExecutor: unrestrained relaxation of one PDB."""
    pdb_in, pdb_out, platform = args
    fname = Path(pdb_in).name
    if Path(pdb_out).exists():
        return {"file": fname, "status": "SKIPPED"}

    try:
        src_dir = str(Path(__file__).resolve().parent.parent.parent /
                      "scripts" / "relaxation")
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "relax_nr_mod",
            Path(src_dir) / "02_relax_no_restraints.py",
        )
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        Path(pdb_out).parent.mkdir(parents=True, exist_ok=True)
        plat_name = "CUDA" if platform == "cuda" else "CPU"
        plat_props = {"CudaPrecision": "mixed"} if platform == "cuda" else {}
        prep = mod.prepare_one_no_restraints(pdb_in, pdb_out, (plat_name, plat_props))
        if prep["status"] != "READY":
            return {"file": fname, "status": "ERROR", "msg": prep.get("msg", "prep failed")}
        result = mod.minimize_one(prep)
        return {"file": fname, "status": result["status"],
                "msg": result.get("msg", "")}
    except Exception as e:
        return {"file": fname, "status": "ERROR", "msg": str(e)}


def run_relaxation(
    pdbs: list[Path],
    out_dir: Path,
    mode_label: str,
    worker_fn,
    platform: str,
    n_workers: int,
    logger: logging.Logger,
) -> dict[str, str]:
    """Parallelize relaxation for a list of PDBs. Returns {fname: status}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(p), str(out_dir / p.name), platform) for p in pdbs]
    results: dict[str, str] = {}
    n_ok = n_skip = n_err = 0

    logger.info(f"[relax/{mode_label}] Starting {len(jobs)} jobs with {n_workers} workers")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        future_to_fname = {ex.submit(worker_fn, j): j[0] for j in jobs}
        for fut in as_completed(future_to_fname):
            pdb_in = future_to_fname[fut]
            fname = Path(pdb_in).name
            try:
                r = fut.result()
                status = r["status"]
                if status == "OK":
                    n_ok += 1
                elif status == "SKIPPED":
                    n_skip += 1
                else:
                    n_err += 1
                    logger.warning(f"[relax/{mode_label}] FAILED {fname}: {r.get('msg','')}")
                results[fname] = status
            except Exception as e:
                n_err += 1
                logger.error(f"[relax/{mode_label}] EXCEPTION {fname}: {e}")
                results[fname] = "ERROR"

    logger.info(
        f"[relax/{mode_label}] Done: {n_ok} OK, {n_skip} skipped, {n_err} errors"
    )
    return results


# =============================================================================
# Step 4 — OpenMM feature extraction
# =============================================================================

def _openmm_worker(args: tuple) -> dict:
    """Worker: extract OpenMM features for one relaxed PDB."""
    pdb_path, vhh_chain, ag_chains, label = args
    fname = Path(pdb_path).name
    try:
        src_dir = str(Path(__file__).resolve().parent.parent.parent /
                      "scripts" / "relaxation")
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "feat_mod",
            Path(src_dir) / "03_extract_features.py",
        )
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)

        return mod.extract_features(pdb_path, [vhh_chain], ag_chains, label)
    except Exception as e:
        return {"file": fname, "label": label, "status": f"ERROR: {e}"}


def run_openmm_features(
    relaxed_rest_dir: Path,
    relaxed_unrest_dir: Path,
    pdbs: list[str],
    vhh_chain: str,
    ag_chains_map: dict[str, list[str]],
    out_csv: Path,
    n_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract OpenMM features for both relax modes. Returns merged DataFrame."""
    from feature_combine import OPENMM_COLS_27 as OMM_COLS  # noqa: F841

    def _run_mode(pdb_dir: Path, suffix: str) -> pd.DataFrame:
        jobs = []
        for fname in pdbs:
            p = pdb_dir / fname
            if not p.exists():
                continue
            ags = ag_chains_map.get(fname, [])
            jobs.append((str(p), vhh_chain, ags, 1))  # label placeholder

        logger.info(f"[openmm/{suffix}] {len(jobs)} jobs, {n_workers} workers")
        rows = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_openmm_worker, j) for j in jobs]
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    rows.append({"file": "?", "status": f"ERROR: {e}"})

        df = pd.DataFrame(rows)
        return df

    rest_df = _run_mode(relaxed_rest_dir, "rest")
    unrest_df = _run_mode(relaxed_unrest_dir, "unrest")

    # Rename unrest columns
    rename_map = {c: f"{c}__unrest" for c in OMM_COLS}
    rename_map["status"] = "status_unrest"
    unrest_renamed = unrest_df.rename(columns=rename_map)

    merged = rest_df.merge(
        unrest_renamed[["file", "status_unrest"] + [f"{c}__unrest" for c in OMM_COLS]],
        on="file", how="outer",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    logger.info(f"[openmm] wrote {out_csv} ({len(merged)} rows)")
    return merged


# =============================================================================
# Step 5 — COCaDA feature extraction
# =============================================================================

def _cocada_worker(args: tuple) -> dict:
    """Worker: extract COCaDA contact features for one PDB."""
    pdb_path, vhh_chain, ag_chains = args
    fname = Path(pdb_path).name
    cocada = Path(COCADA_PATH).resolve()
    if str(cocada) not in sys.path:
        sys.path.insert(0, str(cocada))
    try:
        from src.parser import parse_pdb
        from src.classes import ProcessingContext
        from src.contacts import contact_detection, change_protonation
        from src.conditions import categories
        from collections import Counter

        protein, ph = parse_pdb(str(pdb_path))
        all_chains = list(set([vhh_chain]) | set(ag_chains))
        uncertainty_flags, local_ct = change_protonation(7.4, silent=True)
        context = ProcessingContext(
            distances=categories,
            modified_distances=False,
            ph=7.4,
            interchain=True,
            chains=all_chains,
            silent=True,
        )
        contacts, _, _ = contact_detection(protein, context, uncertainty_flags, local_ct)
        vhh_set = {vhh_chain}
        ag_set = set(ag_chains)
        iface = [c for c in contacts
                 if (c.chain1 in vhh_set and c.chain2 in ag_set) or
                    (c.chain1 in ag_set and c.chain2 in vhh_set)]
        tc = Counter(c.type for c in iface)
        return {
            "file": fname,
            "n_salt_bridges_int": tc.get("salt_bridge", 0) + tc.get("uncertain_salt_bridge", 0),
            "n_aromatic_stacking_int": sum(tc.get(t, 0) for t in
                                           ["stacking-parallel", "stacking-perpendicular", "stacking-other"]),
            "n_hydrophobic_int": tc.get("hydrophobic", 0),
            "n_repulsive_int": tc.get("repulsive", 0) + tc.get("uncertain_repulsive", 0),
            "status": "OK",
        }
    except Exception as e:
        return {"file": fname, "status": f"ERROR: {e}",
                "n_salt_bridges_int": None, "n_aromatic_stacking_int": None,
                "n_hydrophobic_int": None, "n_repulsive_int": None}


def run_cocada_features(
    relaxed_rest_dir: Path,
    relaxed_unrest_dir: Path,
    pdbs: list[str],
    vhh_chain: str,
    ag_chains_map: dict[str, list[str]],
    out_csv: Path,
    n_workers: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract COCaDA features for both relax modes. Returns merged DataFrame."""
    from feature_combine import COCADA_COLS_4

    def _run_mode(pdb_dir: Path) -> pd.DataFrame:
        jobs = []
        for fname in pdbs:
            p = pdb_dir / fname
            if not p.exists():
                continue
            ags = ag_chains_map.get(fname, [])
            jobs.append((str(p), vhh_chain, ags))

        rows = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_cocada_worker, j) for j in jobs]
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    rows.append({"file": "?", "status": f"ERROR: {e}",
                                 "n_salt_bridges_int": None,
                                 "n_aromatic_stacking_int": None,
                                 "n_hydrophobic_int": None,
                                 "n_repulsive_int": None})
        return pd.DataFrame(rows)

    rest_df = _run_mode(relaxed_rest_dir)
    unrest_df = _run_mode(relaxed_unrest_dir)

    rename_map = {c: f"{c}__unrest" for c in COCADA_COLS_4}
    rename_map["status"] = "status_cocada_unrest"
    unrest_renamed = unrest_df.rename(columns=rename_map)

    merged = rest_df.merge(
        unrest_renamed[["file", "status_cocada_unrest"] +
                       [f"{c}__unrest" for c in COCADA_COLS_4]],
        on="file", how="outer",
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    logger.info(f"[cocada] wrote {out_csv} ({len(merged)} rows)")
    return merged


# =============================================================================
# Step 6 — ESM-2 feature extraction
# =============================================================================

def _extract_atom_sequences(pdb_path: str) -> dict[str, str]:
    """Extract per-chain sequences from ATOM records."""
    chains: dict[str, list] = {}
    seen: dict = {}
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            ch = line[21]
            resname = line[17:20].strip()
            resseq = int(line[22:26])
            icode = line[26]
            key = (ch, resseq, icode)
            if key in seen:
                continue
            seen[key] = True
            if ch not in chains:
                chains[ch] = []
            chains[ch].append((resseq, AA3TO1.get(resname, "X")))
    return {ch: "".join(aa for _, aa in sorted(residues))
            for ch, residues in chains.items()}


def run_esm_features(
    relaxed_rest_dir: Path,
    pdbs: list[str],
    vhh_chain: str,
    ag_chains_map: dict[str, list[str]],
    out_csv: Path,
    device_str: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Extract ESM-PPI embeddings. Runs on a single device (no multiprocessing)."""
    esm_ckpt = Path(ESM_CKPT)
    if not esm_ckpt.exists():
        logger.error(f"[esm] ESM-PPI checkpoint not found: {esm_ckpt}")
        logger.error("[esm] Download with: python scripts/download_assets.py --esm")
        return pd.DataFrame()

    try:
        import torch
        from IgGM.model.arch.core.ppi_model import PPIModel

        device = torch.device(device_str)
        logger.info(f"[esm] Loading PPIModel from {esm_ckpt} onto {device}")
        model = PPIModel.restore(str(esm_ckpt))
        model.eval()
        model = model.to(device)

        cols = ["file"]
        cols += [f"vhh_embed_{i}" for i in range(ESM_VHH_DIM)]
        cols += [f"ag_embed_{i}" for i in range(ESM_AG_DIM)]
        cols += [f"cross_attn_{i}" for i in range(ESM_CA_DIM)]

        rows = []
        for fname in pdbs:
            p = relaxed_rest_dir / fname
            if not p.exists():
                logger.warning(f"[esm] SKIP (no rest PDB): {fname}")
                continue
            try:
                atom_seqs = _extract_atom_sequences(str(p))
                if vhh_chain not in atom_seqs:
                    logger.warning(f"[esm] no chain {vhh_chain} in {fname}")
                    continue
                vhh_seq = atom_seqs[vhh_chain]
                ag_chains = ag_chains_map.get(fname, [])
                antigen_seqs = [(ch, atom_seqs[ch]) for ch in ag_chains if ch in atom_seqs]
                if not antigen_seqs:
                    logger.warning(f"[esm] no antigen chains in {fname}")
                    continue

                sequences = [vhh_seq] + [s for _, s in antigen_seqs]
                L_vhh = len(vhh_seq)
                with torch.no_grad():
                    out = model(sequences, mask_prob=0.0)
                sfea = out["sfea"][0].cpu().numpy()
                pfea = out["pfea"][0].cpu().numpy()
                vhh_embed = sfea[:L_vhh].mean(axis=0)
                ag_embed = sfea[L_vhh:].mean(axis=0)
                cross_block = pfea[:L_vhh, L_vhh:]
                cross_attn = cross_block.mean(axis=(0, 1))

                row = [fname]
                row.extend(vhh_embed.tolist())
                row.extend(ag_embed.tolist())
                row.extend(cross_attn.tolist())
                rows.append(row)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                try:
                    logger.warning(f"[esm] OOM on GPU for {fname} — retrying on CPU")
                    model_cpu = model.cpu()
                    atom_seqs = _extract_atom_sequences(str(p))
                    vhh_seq = atom_seqs[vhh_chain]
                    antigen_seqs = [(ch, atom_seqs[ch]) for ch in ag_chains_map.get(fname, []) if ch in atom_seqs]
                    sequences = [vhh_seq] + [s for _, s in antigen_seqs]
                    L_vhh = len(vhh_seq)
                    with torch.no_grad():
                        out = model_cpu(sequences, mask_prob=0.0)
                    sfea = out["sfea"][0].cpu().numpy()
                    pfea = out["pfea"][0].cpu().numpy()
                    vhh_embed = sfea[:L_vhh].mean(axis=0)
                    ag_embed = sfea[L_vhh:].mean(axis=0)
                    cross_attn = pfea[:L_vhh, L_vhh:].mean(axis=(0, 1))
                    row = [fname] + vhh_embed.tolist() + ag_embed.tolist() + cross_attn.tolist()
                    rows.append(row)
                    model.to(device)
                except Exception as e2:
                    model.to(device)
                    logger.error(f"[esm] CPU fallback failed for {fname}: {e2}")
                continue
            except Exception as e:
                logger.error(f"[esm] ERROR {fname}: {e}")
                continue

        if rows:
            df = pd.DataFrame(rows, columns=cols)
        else:
            df = pd.DataFrame(columns=cols)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"[esm] wrote {out_csv} ({len(df)} rows)")
        return df

    except ImportError as e:
        logger.error(f"[esm] PPIModel import failed: {e}")
        logger.error("[esm] ESM features will be skipped.")
        return pd.DataFrame()


# =============================================================================
# Step 7 — Feature assembly
# =============================================================================

def _detect_antigen_chains(pdb_path: str, vhh_chain: str) -> list[str]:
    """Detect antigen chain IDs by reading ATOM records (all chains except VHH)."""
    chains_seen: list[str] = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and len(line) > 21:
                ch = line[21]
                if ch not in chains_seen and ch != vhh_chain and ch.strip():
                    chains_seen.append(ch)
    return chains_seen


def assemble_features(
    openmm_df: pd.DataFrame,
    cocada_df: pd.DataFrame,
    esm_df: pd.DataFrame,
    training_data_dir: Path,
    logger: logging.Logger,
) -> pd.DataFrame | None:
    """Merge all feature blocks and apply ESM PCA using training cohort."""
    from feature_combine import OPENMM_COLS_27, COCADA_COLS_4, esm_pca_cols, unrest_suffix

    # Inner-join all feature sources on 'file'
    merged = openmm_df.copy()
    if not cocada_df.empty:
        merged = merged.merge(cocada_df[["file"] + COCADA_COLS_4 +
                                        [f"{c}__unrest" for c in COCADA_COLS_4]].copy(),
                              on="file", how="inner")
    else:
        logger.warning("[assemble] no COCaDA features — columns will be NaN")
        for c in COCADA_COLS_4:
            merged[c] = float("nan")
            merged[f"{c}__unrest"] = float("nan")

    esm_feat_cols = [c for c in esm_df.columns if c != "file"] if not esm_df.empty else []
    if not esm_df.empty:
        merged = merged.merge(esm_df[["file"] + esm_feat_cols], on="file", how="inner")
    else:
        logger.warning("[assemble] no ESM features — ESM PCA will be NaN")
        esm_feat_cols = []

    # ── ESM PCA ─────────────────────────────────────────────────────────────
    # Fit PCA on the training cohort to match what the models expect.
    # The training cohort ESM CSVs must be reachable.
    n_esm = 64
    pca_cols = esm_pca_cols(n_esm)
    if esm_feat_cols:
        try:
            from sklearn.decomposition import PCA

            # Load training ESM (positives + negatives combined)
            tr_esm_pos = training_data_dir / "esm_ppi_positives.csv"
            tr_esm_neg = training_data_dir / "esm_ppi_negatives.csv"

            if tr_esm_pos.exists() and tr_esm_neg.exists():
                tr_esm = pd.concat([
                    pd.read_csv(tr_esm_pos),
                    pd.read_csv(tr_esm_neg),
                ], ignore_index=True)
                tr_esm_feats = [c for c in tr_esm.columns if c != "file"]
                Xtr = tr_esm[tr_esm_feats].values.astype(np.float32)
                mean = Xtr.mean(axis=0)
                std = Xtr.std(axis=0) + 1e-8
                Xs_tr = (Xtr - mean) / std
                pca = PCA(n_components=n_esm, random_state=42)
                pca.fit(Xs_tr)

                # Project inference ESM features
                Xinf = merged[esm_feat_cols].values.astype(np.float32)
                Xs_inf = (Xinf - mean) / std
                Z = pca.transform(Xs_inf).astype(np.float32)

                for i, pc in enumerate(pca_cols):
                    merged[pc] = Z[:, i]
                merged.drop(columns=esm_feat_cols, inplace=True)
                logger.info(f"[assemble] ESM PCA fit on {len(Xtr)} training samples → {n_esm} dims")
            else:
                logger.warning(
                    "[assemble] Training ESM CSVs not found — fitting PCA on inference data only "
                    f"(looked for {tr_esm_pos}). Scores may differ from training calibration."
                )
                Xinf = merged[esm_feat_cols].values.astype(np.float32)
                mean = Xinf.mean(axis=0)
                std = Xinf.std(axis=0) + 1e-8
                Xs = (Xinf - mean) / std
                pca = PCA(n_components=min(n_esm, Xinf.shape[0] - 1, Xinf.shape[1]),
                          random_state=42)
                Z = pca.fit_transform(Xs).astype(np.float32)
                # Pad to n_esm dims if needed
                if Z.shape[1] < n_esm:
                    pad = np.zeros((Z.shape[0], n_esm - Z.shape[1]), dtype=np.float32)
                    Z = np.concatenate([Z, pad], axis=1)
                for i, pc in enumerate(pca_cols):
                    merged[pc] = Z[:, i]
                merged.drop(columns=esm_feat_cols, inplace=True)
        except Exception as e:
            logger.error(f"[assemble] ESM PCA failed: {e}")
            for pc in pca_cols:
                merged[pc] = float("nan")
            if esm_feat_cols:
                merged.drop(columns=[c for c in esm_feat_cols if c in merged.columns], inplace=True)
    else:
        for pc in pca_cols:
            merged[pc] = float("nan")

    return merged


# =============================================================================
# Step 8 — Scoring
# =============================================================================

def resolve_device(device_arg: str, mode: str, logger: logging.Logger) -> str:
    """Resolve the effective device string for MLP inference."""
    if mode == "rf":
        logger.info("[device] RF runs on CPU only")
        return "cpu"

    if device_arg == "cpu":
        return "cpu"
    elif device_arg == "gpu":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("[device] --device gpu requested but CUDA unavailable — falling back to CPU")
                return "cpu"
            return "cuda"
        except ImportError:
            logger.warning("[device] PyTorch not available — falling back to CPU")
            return "cpu"
    else:  # auto
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("[device] auto → CUDA")
                return "cuda"
        except ImportError:
            pass
        logger.info("[device] auto → CPU")
        return "cpu"


def score_rf(
    checkpoint_dir: Path,
    X: np.ndarray,
    logger: logging.Logger,
) -> np.ndarray:
    """Score with RF. Returns probability array (n_samples,)."""
    import joblib
    import warnings
    model_path = checkpoint_dir / "rf_both_all" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"RF checkpoint not found: {model_path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = joblib.load(model_path)
    logger.info(f"[rf] Loaded checkpoint: {model_path}")
    probs = pipe.predict_proba(X)[:, 1]
    logger.info(f"[rf] Scored {len(probs)} samples")
    return probs


def score_mlp(
    checkpoint_dir: Path,
    X: np.ndarray,
    device_str: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Score with MLP. Returns probability array (n_samples,)."""
    import torch
    import torch.nn as nn

    model_path = checkpoint_dir / "mlp_both_all" / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"MLP checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    mode = ckpt["mode"]            # "both_all"
    hidden = ckpt["hidden_dims"]   # [256, 128]
    dropout = ckpt["dropout"]

    # Reconstruct SingleBranchMLP (both_all uses assemble_matrix flat path).
    # The checkpoint was saved from SingleBranchMLP which wraps layers in self.net,
    # so state_dict keys are "net.0.weight" etc.  We must match that structure.
    class _SingleBranchMLP(nn.Module):
        def __init__(self, n_features: int, hidden_dims: list, drop: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = n_features
            for h in hidden_dims:
                layers.extend([
                    nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(drop),
                ])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            return self.net(x).squeeze(-1)

    n_in = X.shape[1]
    model = _SingleBranchMLP(n_in, hidden, dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Apply stored scaler
    scalers = ckpt.get("scalers", {})
    scale_info = scalers.get("x")
    if scale_info is not None:
        mean = np.asarray(scale_info["mean"], dtype=np.float32)
        scale = np.asarray(scale_info["scale"], dtype=np.float32)
        X = (X - mean) / scale

    device = torch.device(device_str)
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(Xt).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()

    logger.info(f"[mlp] Loaded checkpoint: {model_path}, device={device_str}, scored {len(probs)} samples")
    return probs


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    args = parse_args()
    t_start = time.time()

    # ── Apply CLI path overrides (take precedence over env vars / defaults) ──
    global ESM_CKPT, COCADA_PATH
    if args.esm_checkpoint is not None:
        ESM_CKPT = str(args.esm_checkpoint.resolve())
    if args.cocada_path is not None:
        COCADA_PATH = str(args.cocada_path.resolve())

    # ── Setup output dirs ────────────────────────────────────────────────────
    out_root = args.output_dir.resolve()
    inter_dir = out_root / "intermediates"
    rest_dir  = inter_dir / "relaxed_rest"
    unrest_dir = inter_dir / "relaxed_unrest"
    out_root.mkdir(parents=True, exist_ok=True)
    inter_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "score.log"
    logger = setup_logging(log_path)
    logger.info(f"=== OpenBinder score.py started ===")
    logger.info(f"Input dir  : {args.input_dir}")
    logger.info(f"Output dir : {out_root}")
    logger.info(f"Mode       : {args.mode}")
    logger.info(f"Device     : {args.device}")

    # ── Step 1: Validate input ───────────────────────────────────────────────
    input_pdbs = sorted(args.input_dir.glob("*.pdb"))
    if not input_pdbs:
        logger.error(f"No *.pdb files found in {args.input_dir}. Exiting.")
        sys.exit(1)
    logger.info(f"[step1] Found {len(input_pdbs)} PDB files")

    n_workers = args.workers or cpu_count()
    vhh_chain = args.vhh_chain

    # ── Detect antigen chains for each PDB ──────────────────────────────────
    ag_chains_map: dict[str, list[str]] = {}
    for pdb in input_pdbs:
        ags = _detect_antigen_chains(str(pdb), vhh_chain)
        ag_chains_map[pdb.name] = ags
        if not ags:
            logger.warning(f"[step1] No antigen chains detected in {pdb.name} "
                           f"(VHH chain={vhh_chain})")

    pdb_names = [p.name for p in input_pdbs]

    # ── Checkpoint/skip tracking ─────────────────────────────────────────────
    failed: dict[str, str] = {}   # fname → reason

    # ── Step 2: Restrained relaxation ───────────────────────────────────────
    if args.skip_relaxation and rest_dir.exists() and any(rest_dir.glob("*.pdb")):
        logger.info(f"[step2] --skip-relaxation: using existing {rest_dir}")
    else:
        logger.info("[step2] Running restrained relaxation ...")
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        platform = "cuda" if has_cuda else "cpu"
        logger.info(f"[step2] Platform: {platform}")

        rest_results = run_relaxation(
            input_pdbs, rest_dir, "rest",
            _relax_restrained_worker, platform, n_workers, logger,
        )
        for fname, status in rest_results.items():
            if status == "ERROR":
                failed[fname] = "restrained relaxation failed"

    # ── Step 3: Unrestrained relaxation ─────────────────────────────────────
    if args.skip_relaxation and unrest_dir.exists() and any(unrest_dir.glob("*.pdb")):
        logger.info(f"[step3] --skip-relaxation: using existing {unrest_dir}")
    else:
        logger.info("[step3] Running unrestrained relaxation ...")
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        platform = "cuda" if has_cuda else "cpu"

        unrest_results = run_relaxation(
            input_pdbs, unrest_dir, "unrest",
            _relax_unrestrained_worker, platform, n_workers, logger,
        )
        for fname, status in unrest_results.items():
            if status == "ERROR" and fname not in failed:
                failed[fname] = "unrestrained relaxation failed"

    # Update pdb_names to only those that relaxed successfully
    relaxed_rest_names = {p.name for p in rest_dir.glob("*.pdb")} if rest_dir.exists() else set()
    relaxed_unrest_names = {p.name for p in unrest_dir.glob("*.pdb")} if unrest_dir.exists() else set()
    scoreable_names = [n for n in pdb_names
                       if n in relaxed_rest_names and n in relaxed_unrest_names]

    if not scoreable_names:
        logger.error("[step3] No PDBs successfully relaxed in both modes. Cannot proceed.")
        sys.exit(1)
    logger.info(f"[step3] {len(scoreable_names)}/{len(pdb_names)} PDBs relaxed in both modes")

    # ── Step 4: OpenMM features ──────────────────────────────────────────────
    openmm_csv = inter_dir / "features_openmm.csv"
    if openmm_csv.exists():
        logger.info(f"[step4] Loading cached OpenMM features: {openmm_csv}")
        openmm_df = pd.read_csv(openmm_csv)
    else:
        logger.info("[step4] Extracting OpenMM features ...")
        openmm_df = run_openmm_features(
            rest_dir, unrest_dir, scoreable_names, vhh_chain,
            ag_chains_map, openmm_csv, n_workers, logger,
        )

    # ── Step 5: COCaDA features ──────────────────────────────────────────────
    cocada_csv = inter_dir / "features_cocada.csv"
    if cocada_csv.exists():
        logger.info(f"[step5] Loading cached COCaDA features: {cocada_csv}")
        cocada_df = pd.read_csv(cocada_csv)
    else:
        logger.info("[step5] Extracting COCaDA features ...")
        if not Path(COCADA_PATH).exists():
            logger.warning(f"[step5] COCaDA not found at {COCADA_PATH} — COCaDA features will be skipped")
            cocada_df = pd.DataFrame()
        else:
            cocada_df = run_cocada_features(
                rest_dir, unrest_dir, scoreable_names, vhh_chain,
                ag_chains_map, cocada_csv, n_workers, logger,
            )

    # ── Step 6: ESM features ─────────────────────────────────────────────────
    esm_csv = inter_dir / "features_esm.csv"
    effective_device = resolve_device(args.device, args.mode, logger)
    if esm_csv.exists():
        logger.info(f"[step6] Loading cached ESM features: {esm_csv}")
        esm_df = pd.read_csv(esm_csv)
    else:
        logger.info("[step6] Extracting ESM-2 features ...")
        esm_device = effective_device
        esm_df = run_esm_features(
            rest_dir, scoreable_names, vhh_chain, ag_chains_map,
            esm_csv, esm_device, logger,
        )

    # ── Step 7: Feature assembly ─────────────────────────────────────────────
    logger.info("[step7] Assembling feature matrix ...")
    training_data_dir = args.training_data_dir or (PROJECT_ROOT / "data")

    merged_df = assemble_features(
        openmm_df, cocada_df, esm_df, training_data_dir, logger,
    )

    if merged_df is None or merged_df.empty:
        logger.error("[step7] Feature assembly produced empty DataFrame. Cannot score.")
        sys.exit(1)

    # Build feature matrix for both_all mode
    from feature_combine import assemble_matrix
    try:
        X, feature_cols = assemble_matrix(merged_df, "both_all", 64)
        logger.info(f"[step7] Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    except KeyError as e:
        logger.error(f"[step7] assemble_matrix failed: {e}")
        logger.error("[step7] Some feature columns are missing — check intermediate CSVs")
        sys.exit(1)

    scored_files = merged_df["file"].astype(str).tolist()

    # ── Step 8: Scoring ──────────────────────────────────────────────────────
    checkpoint_dir = args.checkpoint_dir or (PROJECT_ROOT / "models" / "checkpoints")
    logger.info(f"[step8] Checkpoint dir: {checkpoint_dir}")

    rf_scores: np.ndarray | None = None
    mlp_scores: np.ndarray | None = None

    if args.mode in ("rf", "both"):
        try:
            rf_scores = score_rf(checkpoint_dir, X, logger)
        except Exception as e:
            logger.error(f"[step8] RF scoring failed: {e}")
            traceback.print_exc()

    if args.mode in ("mlp", "both"):
        try:
            mlp_scores = score_mlp(checkpoint_dir, X, effective_device, logger)
        except Exception as e:
            logger.error(f"[step8] MLP scoring failed: {e}")
            traceback.print_exc()

    # ── Step 9: Write scores.csv ─────────────────────────────────────────────
    result_rows = []
    for i, fname in enumerate(scored_files):
        rf_s = float(rf_scores[i]) if rf_scores is not None else float("nan")
        mlp_s = float(mlp_scores[i]) if mlp_scores is not None else float("nan")

        available = [s for s in [rf_s, mlp_s] if not math.isnan(s)]
        ens_s = float(np.mean(available)) if available else float("nan")

        rf_lbl = ("binder" if rf_s >= 0.5 else "non-binder") if not math.isnan(rf_s) else "N/A"
        mlp_lbl = ("binder" if mlp_s >= 0.5 else "non-binder") if not math.isnan(mlp_s) else "N/A"

        result_rows.append({
            "pdb_name": fname,
            "rf_score": rf_s,
            "mlp_score": mlp_s,
            "ensemble_score": ens_s,
            "rf_label": rf_lbl,
            "mlp_label": mlp_lbl,
        })

    # Add NaN rows for PDBs that failed relaxation / feature extraction
    scored_set = set(scored_files)
    for fname in pdb_names:
        if fname not in scored_set:
            result_rows.append({
                "pdb_name": fname,
                "rf_score": float("nan"),
                "mlp_score": float("nan"),
                "ensemble_score": float("nan"),
                "rf_label": "N/A",
                "mlp_label": "N/A",
            })

    scores_df = pd.DataFrame(result_rows)
    scores_df = scores_df.sort_values("ensemble_score", ascending=False, na_position="last")
    scores_csv = out_root / "scores.csv"
    scores_df.to_csv(scores_csv, index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    n_scored = int(scores_df["ensemble_score"].notna().sum())
    n_nan = int(scores_df["ensemble_score"].isna().sum())
    n_failed = len(failed)
    n_binder_rf = int((scores_df["rf_label"] == "binder").sum()) if rf_scores is not None else 0
    n_binder_mlp = int((scores_df["mlp_label"] == "binder").sum()) if mlp_scores is not None else 0
    elapsed = time.time() - t_start

    summary_lines = [
        "",
        "=== OpenBinder Scoring Summary ===",
        f"Input PDBs       : {len(pdb_names)}",
        f"Scored           : {n_scored}",
        f"NaN / failed     : {n_nan}",
        f"Relaxation errors: {n_failed}",
        f"RF  binders ≥0.5 : {n_binder_rf}" if rf_scores is not None else "RF               : not run",
        f"MLP binders ≥0.5 : {n_binder_mlp}" if mlp_scores is not None else "MLP              : not run",
        f"Elapsed          : {elapsed:.1f}s",
        f"Scores CSV       : {scores_csv}",
        f"Log              : {log_path}",
    ]
    summary = "\n".join(summary_lines)
    logger.info(summary)
    print(summary)


if __name__ == "__main__":
    main()
