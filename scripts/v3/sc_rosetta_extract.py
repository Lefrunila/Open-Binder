#!/usr/bin/env python3
"""sc_rosetta_extract.py — Generate real Rosetta SC reference values via PyRosetta.

Runs InterfaceAnalyzerMover with compute_interface_sc=True on every relaxed
positive PDB and writes (file, sc_rosetta) pairs to
data/sc_rosetta_reference.csv.

The interface string is derived per-PDB: "H_<all_non_H_chains_concatenated>".
For example a 3-chain complex with chains H, A, B becomes "H_AB".

Crash isolation: each PDB is processed in its own subprocess so a PyRosetta
segfault on one structure cannot poison neighbouring workers. Workers are
managed by a thread pool that calls subprocess.run with a per-task timeout.

Usage:
    python scripts/v3/sc_rosetta_extract.py \
        --input-dir /path/to/positives_relaxed \
        --output    data/sc_rosetta_reference.csv \
        --workers   8

    # internal worker mode (one PDB at a time)
    python scripts/v3/sc_rosetta_extract.py --worker <pdb> <iface>
"""
from __future__ import annotations
import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_PATH = os.path.abspath(__file__)
PYTHON_EXE = sys.executable
WORKER_TIMEOUT_S = 120  # generous per-PDB ceiling


def get_interface_chains(pdb_path: str) -> str:
    """Read CA records, return iface string 'H_<rest>' or empty if no H or no antigen."""
    chains = set()
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                chains.add(line[21])
    if 'H' not in chains:
        return ''
    others = sorted(c for c in chains if c != 'H')
    if not others:
        return ''
    return 'H_' + ''.join(others)


def run_worker_inline(pdb_path: str, iface: str) -> None:
    """Worker entry point — one PDB, prints 'OK <sc>' or 'ERR <msg>' on stdout."""
    try:
        import pyrosetta
        pyrosetta.init('-mute all -ignore_unrecognized_res')
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
        pose = pyrosetta.pose_from_pdb(pdb_path)
        iam = InterfaceAnalyzerMover(iface)
        iam.set_compute_packstat(False)
        iam.set_compute_interface_sc(True)
        iam.set_pack_separated(False)
        iam.apply(pose)
        sc = float(iam.get_all_data().sc_value)
        # Use a sentinel prefix so the parent can ignore PyRosetta init banner.
        print(f'__SC_RESULT__ OK {sc:.6f}', flush=True)
    except Exception as e:
        print(f'__SC_RESULT__ ERR {type(e).__name__}: {e}', flush=True)
        sys.exit(0)  # not 1 — ERR was successfully reported


def call_subprocess(pdb_path: str, iface: str) -> tuple[str, float, str]:
    """Run worker as a subprocess; return (sc_str, sc_float_or_nan, note)."""
    fname = os.path.basename(pdb_path)
    if not iface:
        return fname, float('nan'), 'no iface'
    cmd = [PYTHON_EXE, '-u', SCRIPT_PATH, '--worker', pdb_path, iface]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=WORKER_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        return fname, float('nan'), 'TIMEOUT'

    # Look for our sentinel line
    for line in proc.stdout.splitlines():
        if line.startswith('__SC_RESULT__'):
            payload = line[len('__SC_RESULT__'):].strip()
            if payload.startswith('OK '):
                try:
                    sc = float(payload[3:])
                    return fname, sc, ''
                except ValueError:
                    return fname, float('nan'), f'PARSE: {payload}'
            elif payload.startswith('ERR '):
                return fname, float('nan'), payload[4:]

    # No sentinel found → worker crashed (segfault, OOM, etc.) before printing
    rc = proc.returncode
    err_tail = (proc.stderr or '').splitlines()[-3:] if proc.stderr else []
    note = f'CRASH rc={rc}'
    if err_tail:
        note += ' | ' + ' / '.join(err_tail)
    return fname, float('nan'), note


def _load_done(output: str) -> set:
    """Read existing CSV (if any) and return the set of filenames already processed."""
    done = set()
    if not os.path.exists(output):
        return done
    with open(output) as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or 'file' not in reader.fieldnames:
            return done
        for row in reader:
            f = row.get('file', '').strip()
            if f:
                done.add(f)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', required=False)
    ap.add_argument('--output', required=False)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--resume', action='store_true',
                    help='Append to output and skip files already present.')
    ap.add_argument('--worker', nargs=2, metavar=('PDB', 'IFACE'),
                    help='Internal use: run a single PDB and exit.')
    args = ap.parse_args()

    if args.worker is not None:
        run_worker_inline(args.worker[0], args.worker[1])
        return

    if not args.input_dir or not args.output:
        ap.error('--input-dir and --output are required (unless --worker)')

    pdbs = sorted(Path(args.input_dir).glob('*.pdb'))
    print(f'[sc_rosetta_extract] {len(pdbs)} PDBs', flush=True)

    done = _load_done(args.output) if args.resume else set()
    if done:
        print(f'[sc_rosetta_extract] resume: {len(done)} already done', flush=True)

    tasks = [
        (str(p), get_interface_chains(str(p)))
        for p in pdbs
        if p.name not in done
    ]
    print(f'[sc_rosetta_extract] {len(tasks)} PDBs to process', flush=True)

    if args.resume and os.path.exists(args.output):
        out = open(args.output, 'a', newline='')
        w = csv.writer(out)
    else:
        out = open(args.output, 'w', newline='')
        w = csv.writer(out)
        w.writerow(['file', 'sc_rosetta', 'iface', 'note'])
        out.flush()

    t0 = time.time()
    n = 0
    n_total = len(tasks)
    n_nan = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(call_subprocess, p, i): (p, i)
            for p, i in tasks
        }
        for fut in as_completed(futs):
            fname, sc, note = fut.result()
            iface = futs[fut][1]
            sc_str = f'{sc:.6f}' if sc == sc else ''
            if not sc_str:
                n_nan += 1
            w.writerow([fname, sc_str, iface, note])
            out.flush()
            n += 1
            if n % 25 == 0 or n == n_total:
                elapsed = time.time() - t0
                print(f'[sc_rosetta_extract] {n}/{n_total} done '
                      f'nan={n_nan} '
                      f'elapsed={elapsed:.0f}s '
                      f'rate={n/max(elapsed,1):.2f}/s', flush=True)

    out.close()
    elapsed = time.time() - t0
    print(f'[sc_rosetta_extract] DONE: {n} rows in {elapsed:.0f}s, nan={n_nan}', flush=True)


if __name__ == '__main__':
    main()
