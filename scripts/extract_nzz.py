"""Extract the huGO NZZ page-scan zips (``data/nzz_zips/nzz_<year>.zip``) into ``data/nzz_scans``.

The zips were written without the UTF-8 flag, so Python decodes the names as
cp437 ("Z├╝rich" instead of "Zürich"); the names are repaired on the way out.
Zips that are still downloading (no central directory yet) are skipped, and
files that already exist with the right size are not rewritten, so the script
can be re-run while downloads are in progress.

Usage: python scripts/extract_nzz.py [--src data/nzz_zips] [--dest data/nzz_scans] [--years 1780 1781] [--verify]
"""
import argparse
import shutil
import zipfile
from pathlib import Path


def fix_name(name: str) -> str:
    """Undo the cp437 mis-decoding of UTF-8 member names."""
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def extract_zip(zip_path: Path, dest: Path, verify: bool = False) -> tuple:
    """Return (extracted, skipped) file counts for one archive."""
    extracted = skipped = 0
    with zipfile.ZipFile(zip_path) as z:
        if verify and (bad := z.testzip()):
            raise zipfile.BadZipFile(f"{zip_path}: corrupt member {bad}")
        for info in z.infolist():
            if info.is_dir():
                continue
            target = dest / fix_name(info.filename)
            if target.exists() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    return extracted, skipped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("data/nzz_zips"))
    ap.add_argument("--dest", type=Path, default=Path("data/nzz_scans"))
    ap.add_argument("--years", type=int, nargs="*", help="only these years (default: all zips found)")
    ap.add_argument("--verify", action="store_true", help="CRC-check every member first (slow)")
    args = ap.parse_args()

    zips = sorted(args.src.glob("nzz_*.zip"))
    if args.years:
        zips = [z for z in zips if z.stem in {f"nzz_{y}" for y in args.years} or "index" in z.stem]
    if not zips:
        raise SystemExit(f"no nzz_*.zip found in {args.src}")

    args.dest.mkdir(parents=True, exist_ok=True)
    incomplete = []
    for zp in zips:
        if not zipfile.is_zipfile(zp):
            incomplete.append(zp.name)
            print(f"SKIP {zp.name}: not a complete zip (still downloading?)")
            continue
        n_new, n_old = extract_zip(zp, args.dest, args.verify)
        print(f"OK   {zp.name}: {n_new} extracted, {n_old} already present")
    if incomplete:
        print(f"\n{len(incomplete)} incomplete archive(s), re-run once the download is done: {', '.join(incomplete)}")


if __name__ == "__main__":
    main()
