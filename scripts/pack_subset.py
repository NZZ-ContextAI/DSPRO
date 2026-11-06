"""Pack the code and a stratified sample of pages into one tar.gz for upload to the GPU hub.

The bundle keeps the repo-relative paths, so on the hub the scripts run unchanged from the
unpacked folder. Contains: ocr/, scripts/, requirements.txt, data/nzz_scans (index.csv + the
PDFs/txt of the sampled pages).

Usage: python scripts/pack_subset.py [--out data/hub_bundle.tar.gz] [--per-year 5]
"""
import argparse
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pages import SCANS_DIR, sample_pages  # noqa: E402

CODE_DIRS = ["ocr", "scripts"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/hub_bundle.tar.gz"))
    ap.add_argument("--per-year", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pages = sample_pages(SCANS_DIR, args.per_year, args.seed)
    files = [Path("requirements.txt"), SCANS_DIR / "index.csv"]
    for d in CODE_DIRS:
        files += [p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for page in pages:
        files += [page.pdf, page.txt]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f))
    print(f"{len(pages)} pages, {len(files)} files -> {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
