"""Download the Zenodo NZZ black-letter ground truth and verify its md5 checksum.

Record: https://doi.org/10.5281/zenodo.3333627
Usage:  python scripts/download_zenodo.py [--dest data/zenodo] [--no-extract] [--keep-zip]
"""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

RECORD_ID = "3333627"
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=Path("data/zenodo"))
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--keep-zip", action="store_true", help="keep the zip after extraction")
    args = ap.parse_args()

    with urllib.request.urlopen(API_URL) as r:
        record = json.load(r)
    args.dest.mkdir(parents=True, exist_ok=True)

    for f in record["files"]:
        target = args.dest / f["key"]
        out = args.dest / "extracted"
        if target.suffix == ".zip" and out.exists() and not target.exists():
            print(f"SKIP {f['key']}: already extracted to {out} (delete it to re-download)")
            continue
        algo, expected = f["checksum"].split(":")
        assert algo == "md5", f"unexpected checksum type {algo}"
        if not (target.exists() and md5sum(target) == expected):
            print(f"Downloading {f['key']} ({f['size'] / 1e6:.0f} MB) ...")
            urllib.request.urlretrieve(f["links"]["self"], target)
        actual = md5sum(target)
        if actual != expected:
            raise SystemExit(f"md5 mismatch for {target}: {actual} != {expected}")
        print(f"OK  {target}  md5={actual}")

        if not args.no_extract and target.suffix == ".zip":
            with zipfile.ZipFile(target) as z:
                z.extractall(out)
            print(f"Extracted to {out}")
            if not args.keep_zip:
                target.unlink()
                print(f"Removed {target}")


if __name__ == "__main__":
    main()
