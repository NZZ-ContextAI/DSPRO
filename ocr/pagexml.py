"""Parse PAGE XML ground truth and export line images with transcriptions.

Only the text of ``TextLine/TextEquiv`` is used. Word-level text is partly
uncorrected in the Zenodo ground truth (see docs/datasets.md).
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


@dataclass
class Line:
    page_id: str
    year: int
    index: int
    text: str
    bbox: tuple  # (left, top, right, bottom) in pixels


def _bbox(points: str, width: int, height: int) -> tuple:
    """Bounding box of a PAGE polygon ("x,y x,y ..."), clipped to the image."""
    xy = [tuple(int(float(v)) for v in p.split(",")) for p in points.split()]
    xs, ys = [x for x, _ in xy], [y for _, y in xy]
    return (max(min(xs), 0), max(min(ys), 0), min(max(xs), width), min(max(ys), height))


def parse_page(xml_path: Path) -> tuple:
    """Return (image_filename, [Line, ...]) in document order."""
    root = ET.parse(xml_path).getroot()
    page = root.find("p:Page", NS)
    width, height = int(page.get("imageWidth")), int(page.get("imageHeight"))
    page_id = Path(xml_path).stem
    year = int(re.search(r"nzz_(\d{4})\d{4}", page_id).group(1))

    lines = []
    for tl in page.iter(f"{{{NS['p']}}}TextLine"):
        eq = tl.find("p:TextEquiv/p:Unicode", NS)  # direct child only, not Word
        text = (eq.text or "").strip() if eq is not None else ""
        coords = tl.find("p:Coords", NS)
        if not text or coords is None:
            continue
        bbox = _bbox(coords.get("points"), width, height)
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        lines.append(Line(page_id, year, len(lines), text, bbox))
    return page.get("imageFilename"), lines


def export_page(xml_path: Path, img_dir: Path, out_dir: Path) -> int:
    """Write ``<page>_<nnnn>.png`` + ``.gt.txt`` pairs; return the line count."""
    image_name, lines = parse_page(xml_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(Path(img_dir) / image_name) as img:
        for ln in lines:
            stem = out_dir / f"{ln.page_id}_{ln.index:04d}"
            img.crop(ln.bbox).convert("RGB").save(stem.with_suffix(".png"))
            stem.with_suffix(".gt.txt").write_text(ln.text + "\n", encoding="utf-8")
    return len(lines)
