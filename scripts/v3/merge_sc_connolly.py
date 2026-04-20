#!/usr/bin/env python3
"""
Merge sc_connolly values into feature CSVs, replacing the old Rosetta sc column.

Usage:
  python merge_sc_connolly.py \
    --feature-csv features_positives_openmm_v2.csv \
    --sc-csv sc_connolly_positives.csv \
    --output features_positives_openmm_v2.csv  # overwrite in-place

The sc_connolly CSV has columns: file, sc_ses
The feature CSV has a column: file (filename only, no path), sc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge Connolly SC into feature CSV")
    parser.add_argument("--feature-csv", required=True, help="Feature CSV to update")
    parser.add_argument("--sc-csv", required=True, help="SC CSV from sc_connolly batch run")
    parser.add_argument("--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    feat = pd.read_csv(args.feature_csv)
    sc_df = pd.read_csv(args.sc_csv)

    print(f"Feature CSV rows: {len(feat)}")
    print(f"SC CSV rows: {len(sc_df)}")

    # Normalize filenames for matching (basename only)
    sc_df["file_key"] = sc_df["file"].apply(lambda x: Path(x).name)
    feat["file_key"] = feat["file"].apply(lambda x: Path(x).name)

    # Build lookup dict: filename -> sc_ses
    sc_map = dict(zip(sc_df["file_key"], sc_df["sc_ses"]))

    # Merge: replace sc column
    n_matched = 0
    n_missing = 0
    new_sc = []
    for fname in feat["file_key"]:
        if fname in sc_map:
            new_sc.append(sc_map[fname])
            n_matched += 1
        else:
            new_sc.append(float("nan"))
            n_missing += 1

    feat["sc"] = new_sc

    # Drop the helper column
    feat = feat.drop(columns=["file_key"])

    print(f"Matched: {n_matched}")
    print(f"Missing (NaN): {n_missing}")

    n_nan_sc = feat["sc"].isna().sum()
    valid = feat["sc"].dropna()
    print(f"\nSC stats (Connolly):")
    print(f"  Total rows     : {len(feat)}")
    print(f"  Valid (non-NaN): {len(valid)}")
    print(f"  NaN            : {n_nan_sc}")
    if len(valid) > 0:
        print(f"  Mean           : {valid.mean():.4f}")
        print(f"  Std            : {valid.std():.4f}")
        print(f"  Min            : {valid.min():.4f}")
        print(f"  Max            : {valid.max():.4f}")

    feat.to_csv(args.output, index=False)
    print(f"\nSaved updated CSV to {args.output}")


if __name__ == "__main__":
    main()
