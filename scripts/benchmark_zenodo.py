"""Measure NZZ OCR and VLM transcription against the human-corrected Zenodo ground truth.

Only pages present in both huGO and Zenodo (one page per year, 1780-1815) can be scored. This is
the objective answer to "whose transcription is closer to the truth?" and also the yardstick
for checking whether the judge VLM agrees with CER.

Uses a line-matched CER (order-independent): page-level CER mostly measures whether the
column reading order equals the Zenodo line order, not transcription quality.

Usage: python scripts/benchmark_zenodo.py --run Qwen3-VL-8B-Instruct   (run_vlm_ocr.py --zenodo first)
       python scripts/benchmark_zenodo.py                              (NZZ OCR only)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pagexml import parse_page  # noqa: E402
from ocr.pages import SCANS_DIR, zenodo_pages  # noqa: E402
from ocr.text import line_cer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="VLM run name under data/vlm_ocr")
    ap.add_argument("--scans", type=Path, default=SCANS_DIR)
    ap.add_argument("--out", type=Path, default=Path("data/vlm_ocr"))
    args = ap.parse_args()

    gt_dir = next(Path("data/zenodo/extracted").glob("*/xml/NZZ_groundtruth"))
    rows = []
    for page, xml in zenodo_pages(gt_dir, args.scans):
        _, lines = parse_page(xml)
        gt_lines = [l.text for l in lines]
        nzz = page.txt.read_text(encoding="utf-8", errors="replace")
        row = {"page_id": page.page_id, "year": page.year, "line_cer_nzz": line_cer(gt_lines, nzz)}
        if args.run:
            f = args.out / args.run / "pages" / f"{page.page_id}.txt"
            if f.exists():
                vlm = f.read_text(encoding="utf-8")
                row |= {"line_cer_vlm": line_cer(gt_lines, vlm)}
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"{len(df)} pages with human ground truth")
    df["decade"] = df["year"] // 10 * 10
    print(df.drop(columns=["page_id", "year"]).groupby("decade").mean().round(3))
    print("\noverall mean:\n", df.drop(columns=["page_id", "year", "decade"]).mean().round(3).to_string())
    df.to_csv(args.out / "benchmark_zenodo.csv", index=False)
    print(f"\nwrote {args.out / 'benchmark_zenodo.csv'}")


if __name__ == "__main__":
    main()
