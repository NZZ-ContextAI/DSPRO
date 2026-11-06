"""Loaders for the huGO historical NZZ scans (index.csv + seitentext files)."""
import csv
import logging
import re
from datetime import date
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "datum",
    "ausgabe_nr",
    "seite",
    "media_id",
    "document_id",
    "ocr_zeichen",
    "pdf",
)


def page_id_from_filename(filename: str) -> str:
    """Convert a huGO filename ``YYYY-MM-DD_NN-PPPPP[.pdf|.txt]`` to ``nzz_YYYYMMDD_PP``.

    PP is the page number within the issue, zero-padded to 2 digits. The issue
    number (NN) is not part of the id; it is kept as ``issue_nr`` metadata.

    Raises:
        ValueError: if the filename does not match the huGO pattern.
    """
    stem = Path(filename).stem  # strips .pdf or .txt
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})_\d+-(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot parse huGO filename: {filename!r}")
    y, mo, d, pp = m.groups()
    return f"nzz_{y}{mo}{d}_{int(pp):02d}"


def load_seitentext(txt_path: Path) -> str:
    """Read a seitentext file, stripped; return "" (with a warning) if missing."""
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        logger.warning("seitentext file missing: %s", txt_path)
        return ""
    return txt_path.read_text(encoding="utf-8", errors="replace").strip()


def _to_int(value: str | None, default: int = 0) -> int:
    """Parse an int from a CSV cell, falling back to ``default`` when blank/invalid."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_index(index_csv: Path) -> list[dict]:
    """Read index.csv into one dict per page with canonical page_id and typed fields.

    ``pdf_path``/``txt_path`` are returned as given in the index (relative to the
    scans directory); :func:`iter_pages` resolves them. Unparseable rows are
    skipped with a warning.

    Raises:
        ValueError: if required columns are missing from the CSV header.
    """
    rows: list[dict] = []
    with Path(index_csv).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"index.csv missing required columns: {missing}")
        for line_no, row in enumerate(reader, start=2):
            try:
                issue_date = date.fromisoformat(row["datum"].strip())
                pdf = Path(row["pdf"].strip())
                if not pdf.name:
                    raise ValueError("empty pdf field")
                page_id = page_id_from_filename(pdf.name)
            except ValueError as exc:
                logger.warning("index.csv line %d skipped: %s", line_no, exc)
                continue
            rows.append(
                {
                    "page_id": page_id,
                    "issue_date": issue_date,
                    "decade": issue_date.year // 10 * 10,
                    "issue_nr": _to_int(row["ausgabe_nr"]),
                    "page_nr": _to_int(row["seite"]),
                    "pdf_path": pdf,
                    "txt_path": pdf.with_suffix(".txt"),
                    "media_id": row["media_id"].strip(),
                    "document_id": row["document_id"].strip(),
                    "ocr_zeichen": _to_int(row["ocr_zeichen"]),
                }
            )
    return rows


def iter_pages(index_csv: Path, scans_dir: Path) -> Iterator[dict]:
    """Yield index metadata + seitentext per page; skip pages with no pdf and no txt."""
    scans_dir = Path(scans_dir)
    for page in parse_index(index_csv):
        pdf_path = scans_dir / page["pdf_path"]
        txt_path = scans_dir / page["txt_path"]
        if not pdf_path.is_file() and not txt_path.is_file():
            logger.warning("skipping %s: neither pdf nor txt found", page["page_id"])
            continue
        yield {
            **page,
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "seitentext": load_seitentext(txt_path),
        }
