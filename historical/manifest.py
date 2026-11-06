"""Build and load the JSON-lines manifest of historical pages."""
import json
import logging
from datetime import date
from pathlib import Path

from historical.loader import iter_pages

logger = logging.getLogger(__name__)


def build_manifest(index_csv: Path, scans_dir: Path, out_path: Path) -> int:
    """Write one JSON object per page to ``out_path``; return the page count.

    The seitentext itself is not stored, only its character count.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for page in iter_pages(index_csv, scans_dir):
            has_txt = page["txt_path"].is_file()
            record = {
                "page_id": page["page_id"],
                "issue_date": page["issue_date"].isoformat(),
                "decade": page["decade"],
                "issue_nr": page["issue_nr"],
                "page_nr": page["page_nr"],
                "pdf_path": str(page["pdf_path"]),
                "txt_path": str(page["txt_path"]) if has_txt else None,
                "has_txt": has_txt,
                "seitentext_chars": len(page["seitentext"]),
                "media_id": page["media_id"],
                "document_id": page["document_id"],
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    logger.info("wrote %d manifest records to %s", count, out_path)
    return count


def load_manifest(path: Path) -> list[dict]:
    """Read a JSON-lines manifest; ``issue_date`` is parsed back to ``date``."""
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            rec["issue_date"] = date.fromisoformat(rec["issue_date"])
            records.append(rec)
    return records
