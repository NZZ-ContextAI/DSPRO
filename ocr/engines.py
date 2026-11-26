"""Classic OCR engines behind one interface: ``engine.transcribe(image) -> text``.

Every engine takes a PIL image (a page or a horizontal tile of a page) and returns plain text,
one printed line per text line. That makes them interchangeable in the benchmark
(``scripts/ocr_benchmark.py``) and comparable with the VLM transcriptions.

Engine specs (``make_engine``):

    tesseract                       Tesseract, lang Fraktur+deu, automatic page segmentation (psm 3)
    tesseract:psm=6                 any Tesseract option as key=value, comma separated (psm, lang)
    kraken:<model.mlmodel>          Kraken baseline segmentation + recognition with that model
    calamari:<a.ckpt.json,b.ckpt…>  Kraken segmentation, Calamari recognition (several ckpts = voting)

Kraken and Calamari are called through their command line tools, not imported. Both pin their
own torch/tensorflow versions, so they usually live in a separate virtualenv; point to the tools
with ``KRAKEN_BIN`` / ``CALAMARI_BIN`` if they are not on ``PATH``.
"""
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

TESSERACT_LANG = "Fraktur+deu"


class Engine:
    """Base class; ``spec`` is the string the engine was built from, ``name`` a folder-safe id."""

    spec: str = ""

    @property
    def name(self) -> str:
        return "".join(c if c.isalnum() or c in "-." else "_" for c in self.spec).strip("_")

    def transcribe(self, image: Image.Image) -> str:
        raise NotImplementedError

    def settings(self) -> dict:
        """Parameters worth recording next to the output (and in MLflow)."""
        return {}


# ─── Tesseract ────────────────────────────────────────────────────────────────

@dataclass
class TesseractEngine(Engine):
    lang: str = TESSERACT_LANG
    psm: int = 3  # 3 = automatic layout analysis; 6 = one uniform block (reads across columns)

    @property
    def spec(self) -> str:
        return f"tesseract:psm={self.psm}" + ("" if self.lang == TESSERACT_LANG else f",lang={self.lang}")

    def transcribe(self, image: Image.Image) -> str:
        import pytesseract

        return pytesseract.image_to_string(image, lang=self.lang, config=f"--psm {self.psm}").strip()

    def settings(self) -> dict:
        return {"lang": self.lang, "psm": self.psm}


# ─── Kraken / Calamari (command line) ─────────────────────────────────────────

def _run(cmd: list) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n{proc.stderr[-2000:]}")


def kraken_segment(image: Image.Image, kraken_bin: str = None) -> list:
    """Baseline segmentation with Kraken's default model; returns the line polygons in reading order."""
    kraken_bin = kraken_bin or os.environ.get("KRAKEN_BIN", "kraken")
    with tempfile.TemporaryDirectory() as tmp:
        src, out = Path(tmp) / "in.png", Path(tmp) / "seg.json"
        image.save(src)
        _run([kraken_bin, "-i", str(src), str(out), "segment", "-bl"])
        seg = json.loads(out.read_text(encoding="utf-8"))
    return [line["boundary"] for line in seg.get("lines", []) if line.get("boundary")]


def crop_line(image: Image.Image, boundary, pad: int = 2) -> Image.Image:
    """Bounding box of a line polygon, slightly padded and clipped to the image."""
    xs, ys = [p[0] for p in boundary], [p[1] for p in boundary]
    box = (max(min(xs) - pad, 0), max(min(ys) - pad, 0),
           min(max(xs) + pad, image.width), min(max(ys) + pad, image.height))
    return image.crop(box)


@dataclass
class KrakenEngine(Engine):
    model: str
    kraken_bin: str = None

    @property
    def spec(self) -> str:
        return f"kraken:{Path(self.model).stem}"

    def transcribe(self, image: Image.Image) -> str:
        kraken_bin = self.kraken_bin or os.environ.get("KRAKEN_BIN", "kraken")
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in.png", Path(tmp) / "out.txt"
            image.save(src)
            _run([kraken_bin, "-i", str(src), str(out), "segment", "-bl", "ocr", "-m", self.model])
            return out.read_text(encoding="utf-8").strip()

    def settings(self) -> dict:
        return {"model": self.model}


@dataclass
class CalamariEngine(Engine):
    checkpoints: list = field(default_factory=list)
    calamari_bin: str = None
    kraken_bin: str = None

    @property
    def spec(self) -> str:
        stem = Path(self.checkpoints[0]).parent.name or Path(self.checkpoints[0]).stem
        return f"calamari:{stem}" + (f"x{len(self.checkpoints)}" if len(self.checkpoints) > 1 else "")

    def transcribe(self, image: Image.Image) -> str:
        calamari_bin = self.calamari_bin or os.environ.get("CALAMARI_BIN", "calamari-predict")
        lines = kraken_segment(image, self.kraken_bin)
        if not lines:
            return ""
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, boundary in enumerate(lines):
                p = Path(tmp) / f"line_{i:04d}.png"
                crop_line(image, boundary).convert("L").save(p)
                paths.append(p)
            _run([calamari_bin, "--checkpoint", *self.checkpoints, "--data.images", *map(str, paths)])
            preds = [p.with_suffix(".pred.txt") for p in paths]
            return "\n".join(p.read_text(encoding="utf-8").strip() if p.exists() else "" for p in preds).strip()

    def settings(self) -> dict:
        return {"checkpoints": self.checkpoints}


# ─── factory ──────────────────────────────────────────────────────────────────

def make_engine(spec: str) -> Engine:
    kind, _, arg = spec.partition(":")
    if kind == "tesseract":
        opts = dict(kv.split("=", 1) for kv in arg.split(",") if kv)
        return TesseractEngine(lang=opts.get("lang", TESSERACT_LANG), psm=int(opts.get("psm", 3)))
    if kind == "kraken":
        if not arg:
            raise ValueError("kraken needs a model: kraken:<path/to/model.mlmodel>")
        return KrakenEngine(model=arg)
    if kind == "calamari":
        if not arg:
            raise ValueError("calamari needs checkpoints: calamari:<a.ckpt.json,b.ckpt.json>")
        return CalamariEngine(checkpoints=arg.split(","))
    raise ValueError(f"unknown engine {spec!r} (tesseract, kraken:<model>, calamari:<ckpts>)")
