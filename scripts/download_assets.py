#!/usr/bin/env python3
"""
Download pretrained model weights and/or structure tarballs from Google Drive,
and the ESM-PPI checkpoint from Zenodo.

Usage:
    python scripts/download_assets.py --weights                          # download all 6 model weights
    python scripts/download_assets.py --esm                              # download ESM-PPI checkpoint from Zenodo (~650 MB)
    python scripts/download_assets.py --weights --esm                    # weights + ESM checkpoint
    python scripts/download_assets.py --structures all                   # download all 6 structure tarballs
    python scripts/download_assets.py --structures positives_cleaned positives_relaxed
    python scripts/download_assets.py --weights --structures all
    python scripts/download_assets.py --weights --extract                # weights + extract tarballs

Requirements:
    pip install gdown requests
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

try:
    import gdown
except ImportError:
    print("gdown is not installed. Run:  pip install gdown")
    sys.exit(1)

ESM_CHECKPOINT = {
    "url": "https://zenodo.org/records/16909543/files/esm_ppi_650m_ab.pth?download=1",
    "dest": "models/esm/esm_ppi_650m_ab.pth",
    "size_bytes": 2_536_598_756,
}

# ---------------------------------------------------------------------------
# Asset manifest — Google Drive file IDs obtained 2026-04-20
# Folder: https://drive.google.com/drive/folders/19RSgSMNdrlzEbnTD6olrpnzH6t1JxqVr
# ---------------------------------------------------------------------------
ASSETS = {
    "weights": {
        "rf_rest": {
            "id": "1gRNwchvUSwQngAKO1vBZCZj71aOVtGsY",
            "dest": "models/checkpoints/rf_rest/model.joblib",
            "size_bytes": 29425778,
        },
        "rf_unrest": {
            "id": "1bQrlGHe7mSxZvf6ah5sJuDm7LO99qYCf",
            "dest": "models/checkpoints/rf_unrest/model.joblib",
            "size_bytes": 29971378,
        },
        "rf_both_delta": {
            "id": "1JPTGDVm_ftC1EFbd-N0bdPaNsOgRv7W6",
            "dest": "models/checkpoints/rf_both_delta/model.joblib",
            "size_bytes": 25045330,
        },
        "rf_both_raw": {
            "id": "1wEsSw8d0k0eKTqPEaEV_UmSzNBu_JjgT",
            "dest": "models/checkpoints/rf_both_raw/model.joblib",
            "size_bytes": 27792306,
        },
        "rf_both_all": {
            "id": "149HPvURaRMKV9_w9nC4nuXIRameXBccx",
            "dest": "models/checkpoints/rf_both_all/model.joblib",
            "size_bytes": 24727170,
        },
        "mlp_both_all": {
            "id": "1G_Uf8p4RPSDEDf-sz2nDT0GDw2AbLIGJ",
            "dest": "models/checkpoints/mlp_both_all/model.pt",
            "size_bytes": 303943,
        },
        # NOTE: upload esm_pca.joblib to Google Drive and replace "TBD" with
        # the real file ID.  Generate locally with: python scripts/v3/rf_train.py
        # (or mlp_train.py) — datamodule.prepare() now saves it automatically.
        "esm_pca": {
            "id": "1XyCU4pnKl5AhwYax-d1vChSO38guNDSO",
            "dest": "models/checkpoints/esm_pca.joblib",
            "size_bytes": 864818,
        },
    },
    "structures": {
        "positives_cleaned": {
            "id": "1P9EXHS_cqR_CS3_cqlqSUb7NNtZtuVD-",
            "dest": "data/structures/positives_cleaned.tar.gz",
            "size_bytes": 84778014,
            "extract_to": "data/structures/positives/",
            "n_files": 1129,
        },
        "positives_relaxed": {
            "id": "11LaE-rmJxQvqm_P6HggB6hioe6V8TBBM",
            "dest": "data/structures/positives_relaxed.tar.gz",
            "size_bytes": 142279155,
            "extract_to": "data/structures/positives_relaxed/",
            "n_files": 1129,
        },
        "positives_unrest": {
            "id": "1Ipq0xp5rqLZRMjq5_FS-ZJSb9wNIfr6-",
            "dest": "data/structures/positives_unrest.tar.gz",
            "size_bytes": 142104028,
            "extract_to": "data/structures/positives_unrest/",
            "n_files": 1129,
        },
        "negatives_cleaned": {
            "id": "1ap88sAGq1SkAf9wUnNuDCoTEnSJ-KDsY",
            "dest": "data/structures/negatives_cleaned.tar.gz",
            "size_bytes": 168796282,
            "extract_to": "data/structures/negatives/",
            "n_files": 2288,
        },
        "negatives_relaxed": {
            "id": "1ShmM1GCQ7ADY65aNgrf0uLRibVaOM3o-",
            "dest": "data/structures/negatives_relaxed.tar.gz",
            "size_bytes": 281901674,
            "extract_to": "data/structures/negatives_relaxed/",
            "n_files": 2288,
        },
        "negatives_unrest": {
            "id": "17BvzFYKgLPpIqi8euemXCCDHpKSsD8xV",
            "dest": "data/structures/negatives_unrest.tar.gz",
            "size_bytes": 283157329,
            "extract_to": "data/structures/negatives_unrest/",
            "n_files": 2288,
        },
    },
}

# Locate the repo root (directory containing this script's parent "scripts/")
REPO_ROOT = Path(__file__).resolve().parent.parent


def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def download_asset(name: str, info: dict) -> bool:
    """Download a single asset. Returns True on success."""
    dest = REPO_ROOT / info["dest"]
    expected_bytes = info["size_bytes"]

    # Check if already present with correct size
    if dest.exists():
        actual = dest.stat().st_size
        if actual == expected_bytes:
            print(f"  [skip] {name} — already present ({human_size(expected_bytes)})")
            return True
        else:
            print(
                f"  [warn] {name} exists but size mismatch "
                f"({human_size(actual)} vs expected {human_size(expected_bytes)}). Re-downloading."
            )

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={info['id']}"
    print(f"  Downloading {name} ({human_size(expected_bytes)}) …")

    try:
        gdown.download(url, str(dest), quiet=False)
    except Exception as exc:
        print(f"  [error] Download failed for {name}: {exc}")
        return False

    # Verify size
    if not dest.exists():
        print(f"  [error] File not created: {dest}")
        return False

    actual = dest.stat().st_size
    if actual != expected_bytes:
        print(
            f"  [error] Size mismatch for {name}: "
            f"got {human_size(actual)}, expected {human_size(expected_bytes)}"
        )
        return False

    print(f"  [ok] {name} — {human_size(actual)}")
    return True


def extract_tarball(name: str, info: dict) -> bool:
    """Extract a structure tarball into its target subdirectory."""
    dest = REPO_ROOT / info["dest"]
    extract_to = REPO_ROOT / info["extract_to"]

    if not dest.exists():
        print(f"  [skip extract] {name} tarball not found at {dest}")
        return False

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {name} → {info['extract_to']} …")

    try:
        with tarfile.open(dest, "r:gz") as tf:
            tf.extractall(path=extract_to)
    except Exception as exc:
        print(f"  [error] Extraction failed for {name}: {exc}")
        return False

    n_pdbs = len(list(extract_to.glob("*.pdb")))
    expected = info["n_files"]
    if n_pdbs != expected:
        print(f"  [warn] Extracted {n_pdbs} PDB files, expected {expected}")
    else:
        print(f"  [ok] Extracted {n_pdbs} PDB files")
    return True


def download_esm_checkpoint() -> bool:
    """Download the ESM-PPI checkpoint from Zenodo using urllib."""
    import urllib.request
    info = ESM_CHECKPOINT
    dest = REPO_ROOT / info["dest"]

    if dest.exists():
        actual = dest.stat().st_size
        if actual == info["size_bytes"]:
            print(f"  [skip] esm_ppi_650m_ab.pth — already present ({human_size(info['size_bytes'])})")
            return True
        else:
            print(f"  [warn] esm_ppi_650m_ab.pth exists but size mismatch. Re-downloading.")

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading esm_ppi_650m_ab.pth from Zenodo ({human_size(info['size_bytes'])}) …")

    try:
        urllib.request.urlretrieve(info["url"], str(dest))
    except Exception as exc:
        print(f"  [error] Download failed: {exc}")
        return False

    actual = dest.stat().st_size
    if actual != info["size_bytes"]:
        print(f"  [error] Size mismatch: got {human_size(actual)}, expected {human_size(info['size_bytes'])}")
        return False

    print(f"  [ok] esm_ppi_650m_ab.pth — {human_size(actual)}")
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OpenBinder model weights, ESM checkpoint, and/or structure tarballs."
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Download all 6 pretrained model weight files from Google Drive.",
    )
    parser.add_argument(
        "--esm",
        action="store_true",
        help="Download the ESM-PPI checkpoint (esm_ppi_650m_ab.pth, ~2.4 GB) from Zenodo.",
    )
    parser.add_argument(
        "--structures",
        nargs="+",
        metavar="NAME",
        help=(
            "Download structure tarballs. Pass 'all' for all 6, or one or more names: "
            + ", ".join(ASSETS["structures"].keys())
        ),
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="After downloading structure tarballs, extract them into data/structures/<subdir>/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights and not args.esm and not args.structures:
        print("Nothing to do. Use --weights, --esm, and/or --structures. See --help.")
        sys.exit(0)

    errors = []

    # --- ESM checkpoint ---
    if args.esm:
        print("\n=== Downloading ESM-PPI checkpoint from Zenodo ===")
        ok = download_esm_checkpoint()
        if not ok:
            errors.append("esm_ppi_650m_ab.pth")

    # --- weights ---
    if args.weights:
        print("\n=== Downloading model weights ===")
        for name, info in ASSETS["weights"].items():
            ok = download_asset(name, info)
            if not ok:
                errors.append(name)

    # --- structures ---
    if args.structures:
        if args.structures == ["all"]:
            keys = list(ASSETS["structures"].keys())
        else:
            keys = []
            for k in args.structures:
                if k not in ASSETS["structures"]:
                    print(
                        f"Unknown structure name '{k}'. "
                        f"Valid options: {', '.join(ASSETS['structures'].keys())}"
                    )
                    sys.exit(1)
                keys.append(k)

        print(f"\n=== Downloading {len(keys)} structure tarball(s) ===")
        for name in keys:
            info = ASSETS["structures"][name]
            ok = download_asset(name, info)
            if not ok:
                errors.append(name)
            elif args.extract:
                extract_tarball(name, info)

    # --- summary ---
    print()
    if errors:
        print(f"Finished with {len(errors)} error(s): {', '.join(errors)}")
        sys.exit(1)
    else:
        print("All downloads completed successfully.")


if __name__ == "__main__":
    main()
