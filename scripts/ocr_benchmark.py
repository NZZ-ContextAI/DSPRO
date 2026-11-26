"""OCR benchmark: run classic engines on the ground-truth tiles and score every source against gold.

Works on a ground-truth experiment folder (``data/gt_experiment/<exp>/``, built by the VLM
pipeline): ``pages.json``, ``tiles/<tile>.jpg``, ``gold/<tile>.txt`` (hand-corrected; ``*.draft.txt``
are ignored) and ``transcripts/<source>/<page>.json``. Engine output is written in the same
transcript format, so engines and VLMs end up side by side.

    run    transcribe tiles with every --engines spec (resumable; see ocr/engines.py)
    score  score all sources against gold -> scores.csv + summary.md (mean, 95 % bootstrap CI,
           per decade); --mlflow logs one run per source to the MLflow experiment ``ocr``

Examples:
    python scripts/ocr_benchmark.py run --exp data/gt_experiment/default --engines tesseract tesseract:psm=6
    python scripts/ocr_benchmark.py run --engines kraken:models/fraktur.mlmodel --only-gold
    python scripts/ocr_benchmark.py score --exp data/gt_experiment/default --mlflow
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.engines import make_engine  # noqa: E402
from ocr.text import score  # noqa: E402

DEFAULT_EXP = Path("data/gt_experiment/default")
METRICS = ["cer", "wer", "cer_norm", "recall_cer", "precision_cer"]


# ─── experiment folder ────────────────────────────────────────────────────────

def load_pages(exp: Path) -> list:
    path = exp / "pages.json"
    if not path.exists():
        raise SystemExit(f"{path} not found: build the tiles first (scripts/gt_experiment.py transcribe)")
    return json.loads(path.read_text(encoding="utf-8"))


def tile_ids(page: dict) -> list:
    return [f"{page['page_id']}_t{k}" for k in range(page["tiles"])]


def gold_tiles(exp: Path) -> dict:
    """``{tile_id: gold text}`` for every hand-corrected tile."""
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((exp / "gold").glob("*.txt"))
            if not p.name.endswith(".draft.txt")}


def sources(exp: Path) -> list:
    d = exp / "transcripts"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


# ─── run ──────────────────────────────────────────────────────────────────────

def transcribe_page(exp: Path, engine, page: dict) -> dict:
    from PIL import Image

    t0 = time.time()
    texts = [engine.transcribe(Image.open(exp / "tiles" / f"{t}.jpg")) for t in tile_ids(page)]
    return {"page_id": page["page_id"], "year": page["year"], "model": engine.spec,
            "served_by": [engine.spec], "seconds": round(time.time() - t0, 1), "usage": {},
            "settings": engine.settings(), "tile_texts": texts, "text": "\n".join(texts)}


def cmd_run(args) -> None:
    pages = load_pages(args.exp)
    if args.only_gold:
        gold_pages = {t.rsplit("_t", 1)[0] for t in gold_tiles(args.exp)}
        pages = [p for p in pages if p["page_id"] in gold_pages]
    if args.limit:
        pages = pages[:args.limit]
    for spec in args.engines:
        engine = make_engine(spec)
        out = args.exp / "transcripts" / engine.name
        out.mkdir(parents=True, exist_ok=True)
        todo = [p for p in pages if not (out / f"{p['page_id']}.json").exists()]
        print(f"{spec} -> {out}: {len(todo)}/{len(pages)} pages to transcribe")
        for i, page in enumerate(todo, 1):
            try:
                rec = transcribe_page(args.exp, engine, page)
            except Exception as e:  # noqa: BLE001 - log and keep going, the page is retried next run
                with (out / "errors.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"page_id": page["page_id"], "error": repr(e)}) + "\n")
                print(f"  [{i}/{len(todo)}] {page['page_id']}: FAILED {e!r}", flush=True)
                continue
            (out / f"{page['page_id']}.txt").write_text(rec["text"] + "\n", encoding="utf-8")
            (out / f"{page['page_id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            print(f"  [{i}/{len(todo)}] {page['page_id']}: {len(rec['text'])} chars, {rec['seconds']:.1f}s",
                  flush=True)


# ─── score ────────────────────────────────────────────────────────────────────

def score_rows(exp: Path, exclude=()) -> list:
    """One row per (source, gold tile) that the source has transcribed."""
    pages = {p["page_id"]: p for p in load_pages(exp)}
    gold = {t: g for t, g in gold_tiles(exp).items() if t not in set(exclude)}
    rows = []
    for src in sources(exp):
        for tile, ref in gold.items():
            page_id, k = tile.rsplit("_t", 1)
            rec_path = exp / "transcripts" / src / f"{page_id}.json"
            if page_id not in pages or not rec_path.exists():
                continue
            rec = json.loads(rec_path.read_text(encoding="utf-8"))
            texts = rec.get("tile_texts", [])
            if int(k) >= len(texts):
                continue
            year = pages[page_id]["year"]
            rows.append({"source": src, "tile": tile, "year": year, "decade": year // 10 * 10,
                         "seconds_per_page": rec.get("seconds"), "ref_chars": len(ref),
                         **score(ref, texts[int(k)])})
    return rows


def bootstrap_ci(values, n: int = 2000, seed: int = 0, level: float = 0.95) -> tuple:
    """Percentile bootstrap CI of the mean. With ~30 tiles a 1-2 point CER gap can be noise."""
    values = list(values)
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return means[int(lo * (n - 1))], means[int(hi * (n - 1))]


def summarize(rows: list):
    """Per source: mean of every metric, CI of the CER, tile count; plus CER per decade."""
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    # only tiles every source has: otherwise sources are compared on different text
    common = set.intersection(*(set(g["tile"]) for _, g in df.groupby("source")))
    df = df[df["tile"].isin(common)]
    summary = df.groupby("source")[METRICS].mean()
    ci = {s: bootstrap_ci(g["cer"]) for s, g in df.groupby("source")}
    summary["cer_ci_low"] = [ci[s][0] for s in summary.index]
    summary["cer_ci_high"] = [ci[s][1] for s in summary.index]
    summary["tiles"] = df.groupby("source")["tile"].nunique()
    summary["seconds_per_page"] = df.groupby("source")["seconds_per_page"].mean()
    by_decade = df.pivot_table(index="decade", columns="source", values="cer", aggfunc="mean")
    return summary.sort_values("cer"), by_decade


def _md(df, floatfmt: str = "{:.3f}") -> str:
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in zip(df.index, df.astype(object).values):  # object keeps int columns as ints
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def log_mlflow(summary, exp: Path, uri: str, experiment: str = "ocr") -> None:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    for src, row in summary.iterrows():
        with mlflow.start_run(run_name=f"benchmark-{src}"):
            mlflow.log_params({"source": src, "gt_experiment": exp.name, "tiles": int(row["tiles"])})
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "tiles" and v == v})
            mlflow.log_artifact(str(exp / "benchmark" / "summary.md"))


def cmd_score(args) -> None:
    rows = score_rows(args.exp, exclude=args.exclude or ())
    if not rows:
        raise SystemExit(f"no gold tiles with transcriptions in {args.exp}")
    summary, by_decade = summarize(rows)
    out = args.exp / "benchmark"
    out.mkdir(exist_ok=True)
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    text = (f"# OCR benchmark `{args.exp.name}`\n\n"
            f"{int(summary['tiles'].max())} gold tiles common to all {len(summary)} sources. "
            "`cer`/`wer` diplomatic and in order, `cer_norm` after normalisation (long s, hyphenation, "
            "whitespace), `recall_cer` = missing/misread text, `precision_cer` = invented text. "
            "Overlapping CIs mean the difference is not established.\n\n"
            f"{_md(summary)}\n\n## CER per decade\n\n{_md(by_decade)}\n")
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    if args.mlflow:
        log_mlflow(summary, args.exp, args.mlflow_uri)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--engines", nargs="+", required=True, help="engine specs, see ocr/engines.py")
    r.add_argument("--only-gold", action="store_true", help="only pages that have gold tiles")
    r.add_argument("--limit", type=int, help="first N pages (smoke test)")
    s = sub.add_parser("score")
    s.add_argument("--exclude", nargs="*", help="tile ids to leave out (e.g. a few-shot example)")
    s.add_argument("--mlflow", action="store_true")
    s.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    for p in (r, s):
        p.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    args = ap.parse_args(argv)
    {"run": cmd_run, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
