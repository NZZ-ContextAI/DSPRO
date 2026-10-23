"""Create the decade-stratified page-level split and write data/splits.csv.

Usage:  python scripts/make_split.py [--src data/zenodo/extracted/<dir>] [--out data/splits.csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.split import SEED, decade_table, make_split, write_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/splits.csv"))
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    src = args.src or next(Path("data/zenodo/extracted").glob("*"))

    pages = [p.stem for p in (src / "xml" / "NZZ_groundtruth").glob("*.xml")]
    split = make_split(pages, seed=args.seed)
    write_split(split, args.out)
    print(decade_table(split))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
