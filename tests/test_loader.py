import csv
from datetime import date
from pathlib import Path

import pytest

from historical.loader import (
    REQUIRED_COLUMNS,
    iter_pages,
    load_seitentext,
    page_id_from_filename,
    parse_index,
)
from historical.manifest import build_manifest, load_manifest


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("1780-01-12_01-00003.pdf", "nzz_17800112_03"),
        ("1815-12-31_52-04.txt", "nzz_18151231_04"),
        ("1790-03-07_1-2.pdf", "nzz_17900307_02"),  # single-digit issue/page
        ("1800-02-09_10-11", "nzz_18000209_11"),  # no extension, page > 9
        ("1801-04-02_03-12345.txt", "nzz_18010402_12345"),  # page wider than 2 digits
        ("some/dir/1801-04-02_03-04.pdf", "nzz_18010402_04"),  # path prefix
    ],
)
def test_page_id_from_filename(filename, expected):
    assert page_id_from_filename(filename) == expected


def test_page_id_from_filename_invalid():
    with pytest.raises(ValueError):
        page_id_from_filename("not-a-date.pdf")
    with pytest.raises(ValueError):
        page_id_from_filename("1790-3-7_1-2.pdf")  # month/day must be 2 digits


def test_load_seitentext_missing(tmp_path):
    assert load_seitentext(tmp_path / "nope.txt") == ""


def test_load_seitentext_strips(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("\n  Hallo Welt \n\n", encoding="utf-8")
    assert load_seitentext(p) == "Hallo Welt"


def _write_index(path: Path, rows: list[dict], columns=REQUIRED_COLUMNS) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _row(datum, issue, page, ocr=10):
    return {
        "datum": datum,
        "ausgabe_nr": issue,
        "seite": page,
        "media_id": f"m{issue}{page}",
        "document_id": f"d{issue}{page}",
        "ocr_zeichen": ocr,
        "pdf": f"{datum}_{int(issue):02d}-{int(page):02d}.pdf",
    }


def test_parse_index_columns(tmp_path):
    idx = tmp_path / "index.csv"
    _write_index(idx, [_row("1781-02-03", "4", "1")])
    rows = parse_index(idx)
    assert len(rows) == 1
    r = rows[0]
    for key in (
        "page_id", "issue_date", "decade", "issue_nr", "page_nr",
        "pdf_path", "txt_path", "media_id", "document_id", "ocr_zeichen",
    ):
        assert key in r
    assert r["page_id"] == "nzz_17810203_01"
    assert r["decade"] == 1780


def test_parse_index_missing_column(tmp_path):
    idx = tmp_path / "index.csv"
    _write_index(idx, [], columns=[c for c in REQUIRED_COLUMNS if c != "seite"])
    with pytest.raises(ValueError):
        parse_index(idx)


def test_build_manifest_roundtrip(tmp_path):
    scans = tmp_path / "scans"
    scans.mkdir()
    idx = scans / "index.csv"
    _write_index(
        idx,
        [
            _row("1781-02-03", "4", "1"),  # pdf + txt
            _row("1792-05-06", "7", "2"),  # pdf only
            _row("1803-07-08", "9", "3"),  # neither -> skipped
        ],
    )
    (scans / "1781-02-03_04-01.pdf").write_bytes(b"%PDF")
    (scans / "1781-02-03_04-01.txt").write_text(" Text \n", encoding="utf-8")
    (scans / "1792-05-06_07-02.pdf").write_bytes(b"%PDF")

    assert len(list(iter_pages(idx, scans))) == 2

    out = tmp_path / "out" / "manifest.jsonl"
    assert build_manifest(idx, scans, out) == 2

    recs = load_manifest(out)
    assert len(recs) == 2
    a, b = recs
    assert isinstance(a["issue_date"], date) and a["issue_date"] == date(1781, 2, 3)
    assert isinstance(a["decade"], int) and isinstance(a["issue_nr"], int)
    assert isinstance(a["page_nr"], int)
    assert isinstance(a["pdf_path"], str) and isinstance(a["txt_path"], str)
    assert a["has_txt"] is True and a["seitentext_chars"] == 4
    assert "seitentext" not in a
    assert b["has_txt"] is False and b["txt_path"] is None and b["seitentext_chars"] == 0
