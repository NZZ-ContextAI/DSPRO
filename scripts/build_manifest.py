"""Build the historical page manifest (JSON lines).

Usage:
    python scripts/build_manifest.py \
        --index data/nzz_scans/index.csv \
        --scans data/nzz_scans \
        --out   data/historical/manifest.jsonl
"""
import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical.manifest import build_manifest, load_manifest  # noqa: E402

try:
    from src.config import BASE_DIR  # noqa: E402
except ImportError:  # python-dotenv not installed; fall back to the repo root
    BASE_DIR = str(Path(__file__).resolve().parents[1])

_BASE = Path(BASE_DIR)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, default=_BASE / "data" / "nzz_scans" / "index.csv")
    ap.add_argument("--scans", type=Path, default=_BASE / "data" / "nzz_scans")
    ap.add_argument("--out", type=Path, default=_BASE / "data" / "historical" / "manifest.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    build_manifest(args.index, args.scans, args.out)
    records = load_manifest(args.out)
    decades = Counter(r["decade"] for r in records)

    print(f"Total pages:     {len(records)}")
    print(f"Pages with txt:  {sum(r['has_txt'] for r in records)}")
    print("Decade distribution:")
    for decade in sorted(decades):
        print(f"  {decade}s: {decades[decade]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
