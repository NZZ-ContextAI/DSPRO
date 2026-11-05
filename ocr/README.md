# ocr
Fraktur OCR: image preprocessing, Tesseract baseline, Kraken fine-tuning, second model (Calamari/TrOCR). Experiments are logged to MLflow (experiment `ocr`).

## VLM ground-truth pipeline (huGO scans 1780–1815)
Goal: decide whether the NZZ's own OCR or a VLM transcription is the better ground truth for embeddings/answer checking.

```
data/nzz_zips/nzz_<year>.zip ──extract_nzz.py──▶ data/nzz_scans/   (PDFs + NZZ OCR .txt + index.csv)
                    ──explore_nzz.py──▶ data/exploration/ (coverage, OCR volume, image size)
                    ──run_vlm_ocr.py──▶ data/vlm_ocr/<run>/pages/*.json|txt   (VLM transcription)
                    ──benchmark_zenodo.py▶ CER/WER of NZZ OCR and VLM vs. human Zenodo GT (36 pages)
                    ──judge_ocr.py────▶ data/vlm_ocr/<run>/judge.jsonl (small VLM judge, per tile)
```

```bash
python scripts/extract_nzz.py                       # idempotent; skips zips still downloading
python scripts/explore_nzz.py
python scripts/run_vlm_ocr.py --zenodo              # 36 pages with human ground truth
python scripts/benchmark_zenodo.py --run Qwen3-VL-8B-Instruct
python scripts/run_vlm_ocr.py --per-year 5          # stratified sample, resumable
python scripts/judge_ocr.py --run Qwen3-VL-8B-Instruct
```
Notes: the transcriber needs a *vision*-language model (Qwen-VL family); a text-only LLM cannot read the scan. Pages are cut into horizontal tiles (`--tiles`) to keep each decode short. The judge sees the image and runs both A/B orders; disagreement counts as a tie.

Track a finished run in MLflow (experiment `ocr`, view with `make mlflow`):
```bash
python scripts/log_mlflow.py --run Qwen3-VL-8B-Instruct
```
Logs model/tiles/GPU as params; line-CER (NZZ OCR vs. VLM against Zenodo GT), judge win rates and seconds per page as metrics; `meta.json` and `judge.jsonl` as artifacts. Runs copied back from the hub work the same way.
