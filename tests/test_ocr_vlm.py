import zipfile

import numpy as np
from PIL import Image

from ocr.judge import judge_pair, parse_verdict, split_by_tile
from ocr.pages import split_rows
from ocr.text import cer, line_cer, normalize, wer
from ocr.transcribe import transcribe_page
from scripts.extract_nzz import extract_zip, fix_name


def test_fix_name_repairs_cp437_mojibake():
    assert fix_name("NZZ_Neue_Z├╝rcher_Zeitung/01_Z├╝rich/") == "NZZ_Neue_Zürcher_Zeitung/01_Zürich/"
    assert fix_name("plain/ascii.pdf") == "plain/ascii.pdf"


def test_extract_zip_idempotent(tmp_path):
    zp = tmp_path / "a.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("x/Z├╝rich/p.txt", "hi")
    dest = tmp_path / "out"
    assert extract_zip(zp, dest) == (1, 0)
    assert (dest / "x/Zürich/p.txt").read_text() == "hi"
    assert extract_zip(zp, dest) == (0, 1)


def test_normalize_dehyphenates_and_folds_long_s():
    assert normalize("Die Bil-\nance  ſehr\ngut") == "Die Bilance sehr gut"
    assert normalize("Dank—-\nbarkeit") == "Dankbarkeit"


def test_cer_wer():
    assert cer("abcd", "abcd") == 0
    assert cer("abcd", "abxd") == 0.25
    assert cer("", "") == 0 and cer("", "x") == 1
    assert wer("a b c d", "a x c d") == 0.25
    assert cer("wider-\nlegen", "widerlegen") == 0  # hyphenation is not an error


def test_split_rows_cuts_in_whitespace():
    arr = np.full((300, 50), 255, np.uint8)
    arr[10:140] = 0
    arr[150:290] = 0  # gap at 140-149 lies inside the +-5% window around the even cut at 150
    tiles = split_rows(Image.fromarray(arr), 2)
    assert len(tiles) == 2 and sum(t.height for t in tiles) == 300
    assert 140 <= tiles[0].height <= 150


def test_split_by_tile_handles_different_reading_order():
    col1 = ["Die Stadt Cork hat dem Lord North", "das Bürgerrecht in einer goldnen", "Kapsel überschickt und gedankt"]
    col2 = ["Madrid vom 23. Junii. Ein Nachtrag", "unsrer Zeitung liefert die Artickel", "der Uebergabe der Vestung"]
    # NZZ reads column by column; the VLM reads band by band (first lines of both columns, then the rest)
    nzz = "\n".join(col1 + col2)
    tiles = ["\n".join(col1[:2] + col2[:2]), "\n".join(col1[2:] + col2[2:])]
    got = split_by_tile(nzz, tiles)
    assert got[0].splitlines() == col1[:2] + col2[:2] or set(got[0].splitlines()) == set(col1[:2] + col2[:2])
    assert set(got[1].splitlines()) == set(col1[2:] + col2[2:])


def test_parse_verdict_is_robust():
    assert parse_verdict('{"winner": "B"}') == "B"
    assert parse_verdict('Sure! {"winner": "A"} done') == "A"
    assert parse_verdict("no idea") == "tie"
    assert parse_verdict('{"winner": "C"}') == "tie"


class FakeJudge:
    """Always prefers candidate text containing 'good', whatever its position."""

    def generate(self, prompt, images=(), **_):
        a = prompt.split("=== Candidate A ===\n")[1].split("\n\n=== Candidate B")[0]
        return '{"winner": "%s"}' % ("A" if "good" in a else "B")


def test_judge_pair_cancels_position_bias():
    assert judge_pair(FakeJudge(), None, "good text", "bad text") == "x"
    assert judge_pair(FakeJudge(), None, "bad text", "good text") == "y"
    assert judge_pair(FakeJudge(), None, "good one", "good two") == "tie"  # order-dependent -> tie


class FakeVLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, images=(), max_new_tokens=0):
        self.calls.append(images[0].size)
        return f"tile{len(self.calls)}"


def test_transcribe_page_one_text_per_tile():
    vlm = FakeVLM()
    out = transcribe_page(vlm, Image.new("RGB", (400, 900), "white"), tiles=3, max_side=300)
    assert out == ["tile1", "tile2", "tile3"]
    assert all(max(s) <= 300 for s in vlm.calls)


def test_line_cer_ignores_reading_order():
    lines = ["Madrid, vom 23. Junii. Ein Nachtrag", "unsrer Zeitung liefert die Artickel"]
    swapped = "unsrer Zeitung liefert die Artickel\nMadrid, vom 23. Junii. Ein Nachtrag"
    assert line_cer(lines, swapped) == 0
    assert 0 < line_cer(lines, swapped.replace("Artickel", "Artikel")) < 0.1
