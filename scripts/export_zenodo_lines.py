"""Export line images (*.png) and transcriptions (*.gt.txt) from the Zenodo ground truth.

Usage:  python scripts/export_zenodo_lines.py [--src data/zenodo/extracted/<dir>] [--out data/zenodo/lines]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pagexml import export_page  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/zenodo/lines"))
    args = ap.parse_args()
    src = args.src or next(Path("data/zenodo/extracted").glob("*"))

    total = 0
    xmls = sorted((src / "xml" / "NZZ_groundtruth").glob("*.xml"))
    for xml in xmls:
        total += export_page(xml, src / "img", args.out)
    print(f"{len(xmls)} pages, {total} lines -> {args.out}")


if __name__ == "__main__":
    main()
