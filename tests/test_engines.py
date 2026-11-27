import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

from ocr import engines
from ocr.engines import CalamariEngine, KrakenEngine, TesseractEngine, crop_line, make_engine


def test_make_engine_specs():
    t = make_engine("tesseract")
    assert isinstance(t, TesseractEngine) and t.psm == 3 and t.lang == "Fraktur+deu"
    assert make_engine("tesseract:psm=6,lang=frk").settings() == {"lang": "frk", "psm": 6}
    assert make_engine("kraken:models/fraktur_v2.mlmodel").name == "kraken_fraktur_v2"
    c = make_engine("calamari:fr19/0.ckpt.json,fr19/1.ckpt.json")
    assert isinstance(c, CalamariEngine) and c.name == "calamari_fr19x2"
    for bad in ("kraken", "calamari:", "abbyy"):
        with pytest.raises(ValueError):
            make_engine(bad)


def test_tesseract_engine_passes_lang_and_psm(monkeypatch):
    seen = {}

    def image_to_string(image, lang, config):
        seen.update(lang=lang, config=config)
        return " Zeile eins\nZeile zwei \n"

    monkeypatch.setitem(sys.modules, "pytesseract", types.SimpleNamespace(image_to_string=image_to_string))
    assert TesseractEngine(psm=4).transcribe(Image.new("L", (20, 10))) == "Zeile eins\nZeile zwei"
    assert seen == {"lang": "Fraktur+deu", "config": "--psm 4"}


def _pairs(cmd):
    """(input, output) pairs of a kraken command line."""
    return [(cmd[i + 1], cmd[i + 2]) for i, a in enumerate(cmd) if a == "-i"]


def test_kraken_engine_runs_all_images_in_one_call(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        for k, (_, out) in enumerate(_pairs(cmd)):
            Path(out).write_text(f"Zeile {k}a\nZeile {k}b\n", encoding="utf-8")

    monkeypatch.setattr(engines, "_run", fake_run)
    eng = KrakenEngine(model="m.mlmodel", kraken_bin="kr")
    texts = eng.transcribe_many([Image.new("L", (20, 10)), Image.new("L", (20, 10))])
    assert texts == ["Zeile 0a\nZeile 0b", "Zeile 1a\nZeile 1b"]
    assert len(calls) == 1 and calls[0][:2] == ["kr", "-r"]
    assert calls[0][-5:] == ["segment", "-bl", "ocr", "-m", "m.mlmodel"]
    assert eng.transcribe(Image.new("L", (20, 10))) == "Zeile 0a\nZeile 0b"


def test_calamari_engine_segments_then_predicts_in_line_order(monkeypatch):
    def fake_run(cmd):
        if cmd[-2:] == ["segment", "-bl"]:  # kraken segmentation: image 0 has two lines, image 1 none
            for k, (_, out) in enumerate(_pairs(cmd)):
                lines = [{"boundary": [[0, 0], [10, 0], [10, 4], [0, 4]]},
                         {"boundary": [[0, 5], [10, 5], [10, 9], [0, 9]]}] if k == 0 else []
                Path(out).write_text(json.dumps({"lines": lines}), encoding="utf-8")
        else:  # calamari: one .pred.txt per line image
            images = cmd[cmd.index("--data.images") + 1:]
            for i, img in enumerate(images):
                Path(img).with_suffix(".pred.txt").write_text(f"Zeile {i}\n", encoding="utf-8")

    monkeypatch.setattr(engines, "_run", fake_run)
    eng = CalamariEngine(checkpoints=["a.ckpt.json"], calamari_bin="cp", kraken_bin="kr")
    assert eng.transcribe_many([Image.new("L", (10, 10)), Image.new("L", (10, 10))]) == ["Zeile 0\nZeile 1", ""]


def test_crop_line_is_padded_and_clipped():
    img = Image.new("L", (50, 20))
    assert crop_line(img, [[5, 5], [30, 5], [30, 12], [5, 12]]).size == (29, 11)
    assert crop_line(img, [[0, 0], [60, 0], [60, 30]]).size == (50, 20)
