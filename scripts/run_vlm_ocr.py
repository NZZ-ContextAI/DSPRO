"""Transcribe a stratified sample of NZZ page scans with a local VLM (resumable).

Writes data/vlm_ocr/<run>/pages/<page_id>.json (+ .txt). Re-running skips finished pages.
Full-page VLM decoding is slow (minutes per page on one GPU), so sample, don't transcribe everything.

Usage: python scripts/run_vlm_ocr.py [--model Qwen/Qwen3-VL-8B-Instruct] [--per-year 5 | --zenodo] [--tiles 3]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pages import SCANS_DIR, load_page_image, sample_pages, zenodo_pages  # noqa: E402
from ocr.transcribe import transcribe_page  # noqa: E402
from ocr.vlm import DEFAULT_MODEL, VLM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--load-4bit", action="store_true", help="nf4 quantisation, for ~32B models on 24 GB")
    ap.add_argument("--run", default=None, help="run name (default: model name)")
    ap.add_argument("--scans", type=Path, default=SCANS_DIR)
    ap.add_argument("--out", type=Path, default=Path("data/vlm_ocr"))
    ap.add_argument("--per-year", type=int, default=5)
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--zenodo", action="store_true",
                    help="transcribe the pages that have human Zenodo ground truth (for benchmark_zenodo.py)")
    ap.add_argument("--limit", type=int, help="stop after N pages (smoke test)")
    ap.add_argument("--tiles", type=int, default=3, help="horizontal bands per page")
    ap.add_argument("--max-side", type=int, default=2560)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--repetition-penalty", type=float, default=1.05,
                    help="1.0 = off; a penalty can push the model away from repeating printed spellings")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = args.run or args.model.split("/")[-1]
    out = args.out / run / "pages"
    out.mkdir(parents=True, exist_ok=True)
    if args.zenodo:
        gt_dir = next(Path("data/zenodo/extracted").glob("*/xml/NZZ_groundtruth"))
        pages = [p for p, _ in zenodo_pages(gt_dir, args.scans)]
    else:
        pages = sample_pages(args.scans, args.per_year, args.seed, years=set(args.years or []))
    todo = [p for p in pages if not (out / f"{p.page_id}.json").exists()][: args.limit]
    print(f"{len(pages)} sampled pages, {len(todo)} to do -> {out}")

    import torch

    meta = {"model": args.model, "load_4bit": args.load_4bit, "tiles": args.tiles, "max_side": args.max_side, "repetition_penalty": args.repetition_penalty,
            "selection": "zenodo" if args.zenodo else f"sample_{args.per_year}_per_year",
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    (args.out / run / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    vlm = VLM(args.model, args.load_4bit)
    vlm.repetition_penalty = args.repetition_penalty
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        tiles = transcribe_page(vlm, load_page_image(p.pdf), args.tiles, args.max_side, args.max_new_tokens)
        rec = {"page_id": p.page_id, "year": p.year, "pdf": str(p.pdf), "model": args.model,
               "tiles": args.tiles, "seconds": round(time.time() - t0, 1),
               "tile_texts": tiles, "text": "\n".join(tiles)}
        (out / f"{p.page_id}.txt").write_text(rec["text"] + "\n", encoding="utf-8")
        (out / f"{p.page_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(todo)}] {p.page_id}: {len(rec['text'])} chars, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
