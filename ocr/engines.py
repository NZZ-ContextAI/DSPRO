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
        return self.transcribe_many([image])[0]

    def transcribe_many(self, images: list) -> list:
        """Several images at once; engines with a slow start (model loading) override this."""
        return [self.transcribe(img) for img in images]

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


def kraken_segment(images: list, kraken_bin: str = None) -> list:
    """Baseline segmentation with Kraken's default model, one call for all images.

    Returns one list of line polygons per image, in the order Kraken lists the lines.
    """
    kraken_bin = kraken_bin or os.environ.get("KRAKEN_BIN", "kraken")
    with tempfile.TemporaryDirectory() as tmp:
        pairs = []
        for i, img in enumerate(images):
            src, out = Path(tmp) / f"in_{i:03d}.png", Path(tmp) / f"seg_{i:03d}.json"
            img.save(src)
            pairs += ["-i", str(src), str(out)]
        _run([kraken_bin, "-r", *pairs, "segment", "-bl"])
        segs = [json.loads((Path(tmp) / f"seg_{i:03d}.json").read_text(encoding="utf-8")) for i in range(len(images))]
    return [[line["boundary"] for line in seg.get("lines", []) if line.get("boundary")] for seg in segs]


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

    def transcribe_many(self, images: list) -> list:
        kraken_bin = self.kraken_bin or os.environ.get("KRAKEN_BIN", "kraken")
        with tempfile.TemporaryDirectory() as tmp:
            pairs = []
            for i, img in enumerate(images):
                src, out = Path(tmp) / f"in_{i:03d}.png", Path(tmp) / f"out_{i:03d}.txt"
                img.save(src)
                pairs += ["-i", str(src), str(out)]
            _run([kraken_bin, "-r", *pairs, "segment", "-bl", "ocr", "-m", self.model])
            return [(Path(tmp) / f"out_{i:03d}.txt").read_text(encoding="utf-8").strip() for i in range(len(images))]

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

    def transcribe_many(self, images: list) -> list:
        calamari_bin = self.calamari_bin or os.environ.get("CALAMARI_BIN", "calamari-predict")
        segs = kraken_segment(images, self.kraken_bin)
        with tempfile.TemporaryDirectory() as tmp:
            per_image, paths = [], []
            for i, (img, lines) in enumerate(zip(images, segs)):
                own = []
                for j, boundary in enumerate(lines):
                    p = Path(tmp) / f"img{i:03d}_line{j:04d}.png"
                    crop_line(img, boundary).convert("L").save(p)
                    own.append(p)
                per_image.append(own)
                paths += own
            if paths:
                _run([calamari_bin, "--checkpoint", *self.checkpoints, "--data.images", *map(str, paths)])

            def read(p):
                pred = p.with_suffix(".pred.txt")
                return pred.read_text(encoding="utf-8").strip() if pred.exists() else ""

            return ["\n".join(read(p) for p in own).strip() for own in per_image]

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
