#!/usr/bin/env python3
"""
Batch SC computation with automatic antigen chain detection.
Detects the first non-H chain in each PDB file and uses it as the antigen chain.
Outputs a CSV with columns: file, sc_ses
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


def get_chains_from_pdb(pdb_path: str) -> list[str]:
    """Return ordered unique chain IDs from ATOM records."""
    seen = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                chain = line[21]
                if chain not in seen:
                    seen.append(chain)
    return seen


def worker(args_tuple):
    pdb_path, chain_vhh, kwargs = args_tuple
    from sc_connolly import compute_sc

    # Auto-detect antigen chain: first non-VHH chain in the file
    chains = get_chains_from_pdb(pdb_path)
    ag_chains = [c for c in chains if c != chain_vhh]

    if not ag_chains:
        logger.error("No antigen chain found in %s (chains: %s)", pdb_path, chains)
        return Path(pdb_path).name, float("nan")

    chain_ag = ag_chains[0]

    try:
        sc = compute_sc(pdb_path, chain_vhh=chain_vhh, chain_ag=chain_ag, **kwargs)
    except Exception as exc:
        logger.error("Error processing %s: %s", pdb_path, exc)
        sc = float("nan")

    return Path(pdb_path).name, sc


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Batch SC with auto antigen-chain detection"
    )
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--vhh-chain", default="H", dest="vhh_chain")
    parser.add_argument("--probe-radius", type=float, default=1.4)
    parser.add_argument("--grid-spacing", type=float, default=0.5)
    parser.add_argument("--interface-cutoff", type=float, default=3.5)
    parser.add_argument("--interface-cutoff-prefilter", type=float, default=2.5)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    pdbs = sorted(batch_dir.glob("*.pdb"))
    if not pdbs:
        print(f"No PDB files found in {batch_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdbs)} PDB files in {batch_dir}", flush=True)

    kwargs = dict(
        probe_radius=args.probe_radius,
        grid_spacing=args.grid_spacing,
        interface_cutoff=args.interface_cutoff,
        interface_cutoff_prefilter=args.interface_cutoff_prefilter,
    )

    tasks = [(str(p), args.vhh_chain, kwargs) for p in pdbs]
    results = []
    done = 0

    # Add script directory to sys.path so sc_connolly can be imported
    script_dir = Path(__file__).parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            fname, sc = fut.result()
            results.append((fname, sc))
            done += 1
            if done % 50 == 0 or done == len(pdbs):
                print(f"Progress: {done}/{len(pdbs)} done", flush=True)

    results.sort(key=lambda x: x[0])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "sc_ses"])
        writer.writerows(results)

    n_nan = sum(1 for _, sc in results if sc != sc)  # NaN check
    print(f"Wrote {len(results)} results ({n_nan} NaN) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
