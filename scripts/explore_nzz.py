"""Explore the extracted huGO NZZ scans 1780-1815: coverage, OCR volume/quality proxies, image size.

Writes data/exploration/{summary.csv,overview.png} and prints key numbers.

Usage: python scripts/explore_nzz.py [--scans data/nzz_scans] [--probe 5]
"""
import argparse
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pages import SCANS_DIR  # noqa: E402


def alpha_ratio(text: str) -> float:
    """Share of letters among non-space characters; low values hint at garbled OCR."""
    chars = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in chars) / len(chars) if chars else 0.0


def embedded_size(pdf: Path) -> tuple:
    import pymupdf

    with pymupdf.open(pdf) as doc:
        imgs = doc[0].get_images(full=True)
        if not imgs:
            return (0, 0)
        info = doc.extract_image(imgs[0][0])
        return info["width"], info["height"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scans", type=Path, default=SCANS_DIR)
    ap.add_argument("--probe", type=int, default=5, help="pages per year probed for image size / OCR quality")
    ap.add_argument("--out", type=Path, default=Path("data/exploration"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.scans / "index.csv", dtype={"ausgabe_nr": str})
    df["year"] = df["datum"].str[:4].astype(int)
    df["extracted"] = [(args.scans / p).exists() for p in df["pdf"]]
    print(f"index: {len(df)} pages, {df['datum'].nunique()} issues, {df['year'].min()}-{df['year'].max()}")
    print(f"pages extracted so far: {df['extracted'].sum()} ({df['extracted'].mean():.0%})")
    print("pages/issue:", df.groupby("datum").size().value_counts().to_dict())
    print("ressort:", df["ressort"].value_counts().head(5).to_dict())
    print("ausgabe:", df["ausgabe"].value_counts().to_dict())

    per_year = df.groupby("year").agg(pages=("pdf", "size"), issues=("datum", "nunique"),
                                      extracted=("extracted", "sum"),
                                      ocr_chars_mean=("ocr_zeichen", "mean"),
                                      ocr_chars_median=("ocr_zeichen", "median"),
                                      short_pages=("ocr_zeichen", lambda s: (s < 300).mean()))

    rng = random.Random(0)
    rows = []
    for year, g in df[df["extracted"]].groupby("year"):
        for _, r in g.sample(min(args.probe, len(g)), random_state=rng.randint(0, 10**6)).iterrows():
            pdf = args.scans / r["pdf"]
            w, h = embedded_size(pdf)
            txt = pdf.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
            rows.append({"year": year, "width": w, "height": h, "alpha": alpha_ratio(txt),
                         "mb": pdf.stat().st_size / 1e6})
    if rows:
        probe = pd.DataFrame(rows).groupby("year").mean()
        per_year = per_year.join(probe)
        print(f"\nimage size (mean): {probe['width'].mean():.0f} x {probe['height'].mean():.0f} px, "
              f"{probe['mb'].mean():.2f} MB/page, NZZ-OCR letter ratio {probe['alpha'].mean():.2f}")
    per_year.round(2).to_csv(args.out / "summary.csv")

    gt_xml = next(Path("data/zenodo/extracted").glob("*/xml/NZZ_groundtruth"), None)
    if gt_xml:
        gt_dates = {re.search(r"nzz_(\d{8})", p.stem).group(1) for p in gt_xml.glob("*.xml")}
        gt_dates = {f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in gt_dates if 1780 <= int(d[:4]) <= 1815}
        overlap = gt_dates & set(df["datum"])
        print(f"\nZenodo GT pages 1780-1815: {len(gt_dates)} dates, {len(overlap)} also in huGO "
              f"-> usable for a real CER benchmark: {sorted(overlap)[:5]}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    per_year["pages"].plot.bar(ax=ax[0], title="pages per year", width=0.8)
    per_year["ocr_chars_mean"].plot(ax=ax[1], title="mean NZZ OCR characters per page")
    per_year["short_pages"].plot(ax=ax[2], title="share of pages with <300 OCR chars")
    for a in ax:
        a.set_xlabel("")
    ax[0].tick_params(axis="x", labelsize=6)
    fig.tight_layout()
    fig.savefig(args.out / "overview.png", dpi=130)
    print(f"\nwrote {args.out}/summary.csv and overview.png")


if __name__ == "__main__":
    main()
