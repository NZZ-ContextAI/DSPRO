"""Tesseract Fraktur baseline: PDF page -> raw OCR text, plus MLflow logging.

System requirements (not installable via pip): the ``tesseract`` binary, the
Fraktur tessdata (``Fraktur.traineddata``, e.g. package
``tesseract-ocr-script-frak``) and poppler (``pdftoppm``). Run
``bash scripts/check_deps.sh`` to verify them.
"""

import logging
import time
from pathlib import Path

import mlflow
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

LANG = "Fraktur+deu"
TESSERACT_CONFIG = "--psm 6"


def transcribe_page(pdf_path: Path, dpi: int = 300) -> str:
    """Render PDF page 1 to an image, run Tesseract (Fraktur+deu) and return the raw string."""
    images = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
    return pytesseract.image_to_string(images[0], lang=LANG, config=TESSERACT_CONFIG)


def batch_transcribe(pages: list[dict], scans_dir: Path, dpi: int = 300) -> list[dict]:
    """Transcribe manifest pages.

    Args:
        pages: Manifest dicts (see historical/manifest.py); ``page_id`` and
            ``pdf_path`` (relative to ``scans_dir``) are used.
        scans_dir: Directory the manifest ``pdf_path`` values are relative to.
        dpi: Rendering resolution.

    Returns:
        One dict per page with page_id, pdf_path, hypothesis, elapsed_seconds.
    """
    results = []
    for i, page in enumerate(pages, 1):
        pdf_path = Path(scans_dir) / page["pdf_path"]
        start = time.perf_counter()
        hypothesis = transcribe_page(pdf_path, dpi=dpi)
        elapsed = time.perf_counter() - start
        logger.info("[%d/%d] %s transcribed in %.1fs", i, len(pages), page["page_id"], elapsed)
        results.append(
            {
                "page_id": page["page_id"],
                "pdf_path": str(pdf_path),
                "hypothesis": hypothesis,
                "elapsed_seconds": elapsed,
            }
        )
    return results


def log_baseline_run(
    results: list[dict], mlflow_uri: str, experiment_name: str = "ocr", *, dpi: int = 300
) -> None:
    """Log mean CER, mean WER, page count and dpi as one MLflow run.

    Args:
        results: Per-page dicts that carry ``cer`` and ``wer`` (see ocr.text.evaluate_page).
        mlflow_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        dpi: Rendering resolution used for the run (an optional ``dpi`` key in
            the results takes precedence).
    """
    if not results:
        raise ValueError("results must not be empty")
    n = len(results)
    dpi = results[0].get("dpi", dpi)
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="tesseract-baseline"):
        mlflow.log_params({"page_count": n, "dpi": dpi})
        mlflow.log_metrics(
            {
                "mean_cer": sum(r["cer"] for r in results) / n,
                "mean_wer": sum(r["wer"] for r in results) / n,
            }
        )
