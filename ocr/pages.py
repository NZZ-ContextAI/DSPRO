"""Locate, sample, render and tile the extracted huGO page scans (data/nzz_scans)."""
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SCANS_DIR = Path("data/nzz_scans")


@dataclass
class PageRef:
    page_id: str  # e.g. 1780-01-12_01-00001
    year: int
    pdf: Path

    @property
    def txt(self) -> Path:
        """The NZZ's own OCR text that ships next to every PDF."""
        return self.pdf.with_suffix(".txt")


def load_index(root: Path = SCANS_DIR) -> list:
    with (root / "index.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sample_pages(root: Path = SCANS_DIR, per_year: int = 5, seed: int = 42, min_chars: int = 300,
                 years=None) -> list:
    """Stratified sample: ``per_year`` pages per year among extracted, non-trivial pages."""
    by_year = defaultdict(list)
    for row in load_index(root):
        year = int(row["datum"][:4])
        pdf = root / row["pdf"]
        if (years and year not in years) or int(row["ocr_zeichen"] or 0) < min_chars or not pdf.exists():
            continue
        by_year[year].append(PageRef(pdf.stem, year, pdf))
    rng = random.Random(seed)
    out = []
    for year in sorted(by_year):
        pages = sorted(by_year[year], key=lambda p: p.page_id)
        out += sorted(rng.sample(pages, min(per_year, len(pages))), key=lambda p: p.page_id)
    return out


def load_page_image(pdf: Path, dpi: int = 200) -> Image.Image:
    """The scan embedded in the PDF at native resolution; falls back to rendering."""
    import io

    import pymupdf

    with pymupdf.open(pdf) as doc:
        page = doc[0]
        images = page.get_images(full=True)
        if len(images) == 1:
            return Image.open(io.BytesIO(doc.extract_image(images[0][0])["image"])).convert("RGB")
        pix = page.get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def fit(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so the longer side is at most ``max_side`` (bounds the VLM token count)."""
    scale = max_side / max(img.size)
    if scale >= 1:
        return img
    return img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)


def split_rows(img: Image.Image, n: int, window: float = 0.05) -> list:
    """Cut into ``n`` horizontal bands, each cut at the emptiest pixel row near the even split.

    Cutting in a gap between text lines avoids slicing glyphs; no overlap is needed.
    """
    if n <= 1:
        return [img]
    ink = (np.asarray(img.convert("L")) < 128).sum(axis=1)
    ink = np.convolve(ink, np.ones(9) / 9, mode="same")
    h = len(ink)
    radius = int(h * window)
    cuts = [0]
    for i in range(1, n):
        centre = h * i // n
        lo, hi = max(centre - radius, cuts[-1] + 1), min(centre + radius, h - 1)
        cuts.append(int(lo + np.argmin(ink[lo:hi])) if hi > lo else centre)
    cuts.append(h)
    return [img.crop((0, a, img.width, b)) for a, b in zip(cuts, cuts[1:])]
