from pathlib import Path

import pytest

from ocr.pagexml import export_page, parse_page

ROOT = next(Path("data/zenodo/extracted").glob("*"), None)
PAGES = ["nzz_17800719_0_0_a1_p1_1", "nzz_18790716_0_0_a1_p1_1"]
pytestmark = pytest.mark.skipif(ROOT is None, reason="Zenodo data not downloaded (make zenodo)")


def _xml(name):
    return ROOT / "xml" / "NZZ_groundtruth" / f"{name}.xml"


def test_first_page_lines_and_year():
    _, lines = parse_page(_xml(PAGES[0]))
    assert len(lines) == 76 and all(l.year == 1780 for l in lines)
    assert all(l.text for l in lines)


def test_text_comes_from_line_level():
    _, lines = parse_page(_xml(PAGES[0]))
    # line-level text is a full string, not a single word of a Word element
    assert max(len(l.text.split()) for l in lines) > 1


def test_export_pairs(tmp_path):
    for name in PAGES:
        xml = _xml(name)
        if not xml.exists():
            pytest.skip(f"{name} missing")
        n = export_page(xml, ROOT / "img", tmp_path)
        assert n > 0
    assert len(list(tmp_path.glob("*.png"))) == len(list(tmp_path.glob("*.gt.txt")))
