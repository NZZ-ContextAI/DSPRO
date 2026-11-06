"""Log a finished VLM-OCR run (data/vlm_ocr/<run>) to MLflow, experiment ``ocr``.

Logs parameters (from meta.json / page records), the CER between NZZ OCR and VLM text (a measure of
disagreement, not of quality), judge win rates if judge.jsonl exists, and seconds per page if recorded. Works for runs copied back from the hub.

Usage: python scripts/log_mlflow.py --run Qwen3-VL-8B-Instruct [--name <mlflow run name>]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pages import SCANS_DIR  # noqa: E402
from src.config import MLFLOW_TRACKING_URI  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", help="MLflow run name (default: --run)")
    ap.add_argument("--out", type=Path, default=Path("data/vlm_ocr"))
    ap.add_argument("--scans", type=Path, default=SCANS_DIR)
    ap.add_argument("--experiment", default="ocr")
    ap.add_argument("--tracking-uri", default=MLFLOW_TRACKING_URI)
    args = ap.parse_args()

    run_dir = args.out / args.run
    recs = {r["page_id"]: r for r in (json.loads(p.read_text(encoding="utf-8"))
                                      for p in sorted((run_dir / "pages").glob("*.json")))}
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    first = next(iter(recs.values()))
    params = {"model": first["model"], "tiles": first["tiles"], "n_pages": len(recs), **meta}
    metrics = {}

    secs = [r["seconds"] for r in recs.values() if "seconds" in r]
    if secs:
        metrics["sec_per_page"] = sum(secs) / len(secs)

    judge_path = run_dir / "judge.jsonl"
    if judge_path.exists():
        wins, cer_pairs = Counter(), []
        for line in judge_path.open(encoding="utf-8"):
            j = json.loads(line)
            wins.update(j["verdicts"])
            cer_pairs += [c for c in j["cer_nzz_vs_vlm"] if c is not None]
        n = sum(v for k, v in wins.items() if k != "skip") or 1
        metrics |= {f"judge_{k}_rate": wins[k] / n for k in ("nzz", "vlm", "tie")}
        metrics["cer_nzz_vs_vlm"] = sum(cer_pairs) / max(len(cer_pairs), 1)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.name or args.run):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        for f in (meta_path, judge_path):
            if f.exists():
                mlflow.log_artifact(str(f), artifact_path="vlm_ocr")
    print(json.dumps({"params": params, "metrics": metrics}, indent=1, default=str))


if __name__ == "__main__":
    main()
