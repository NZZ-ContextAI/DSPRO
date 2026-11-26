from PIL import Image

from ocr import tesseract


def test_transcribe_page_calls_tesseract(tmp_path, monkeypatch):
    seen = {}

    def fake_convert(path, **kwargs):
        seen["convert"] = kwargs
        return [Image.new("RGB", (10, 10), "white")]

    def fake_ocr(image, **kwargs):
        seen["ocr"] = kwargs
        return "test output"

    monkeypatch.setattr(tesseract, "convert_from_path", fake_convert)
    monkeypatch.setattr(tesseract.pytesseract, "image_to_string", fake_ocr)

    assert tesseract.transcribe_page(tmp_path / "page.pdf") == "test output"
    assert seen["convert"]["first_page"] == 1 and seen["convert"]["last_page"] == 1
    assert seen["ocr"] == {"lang": "Fraktur+deu", "config": "--psm 6"}
