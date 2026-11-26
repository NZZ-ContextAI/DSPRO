# ocr
Fraktur OCR: image preprocessing, Tesseract baseline, Kraken fine-tuning, second model (Calamari/TrOCR). Experiments are logged to MLflow (experiment `ocr`).

## Benchmark: classic engines vs. VLMs on the gold tiles (`scripts/ocr_benchmark.py`)
Engines (`ocr/engines.py`) share one interface, `engine.transcribe(image) -> text`:
`tesseract` (Fraktur+deu, `psm=3` default), `kraken:<model.mlmodel>`, `calamari:<ckpt,…>` (Kraken
segmentation + Calamari lines). Kraken/Calamari run via their CLIs, so they can live in their own venv
(`KRAKEN_BIN`, `CALAMARI_BIN`); `bash scripts/check_deps.sh` lists what is installed.

```bash
python scripts/ocr_benchmark.py run --exp data/gt_experiment/default --engines tesseract tesseract:psm=6 --only-gold
python scripts/ocr_benchmark.py score --exp data/gt_experiment/default --mlflow
```
`run` writes `transcripts/<engine>/<page>.json` in the same format as the VLM transcripts. `score` scores every
source on the gold tiles they all cover -> `benchmark/summary.md` + `scores.csv`: diplomatic `cer`/`wer`,
`cer_norm` (long s, hyphenation, whitespace normalised), order-independent `recall_cer`/`precision_cer`, a 95 %
bootstrap CI of the CER and CER per decade. `ocr.text.cer` is diplomatic by default; pass `norm=True` to normalise.
