"""Grade NZZ OCR vs. VLM transcription per tile with a small judge VLM.

Reads data/vlm_ocr/<run>/pages/*.json, writes data/vlm_ocr/<run>/judge.jsonl and prints the
win rates (overall and per decade). Also reports the CER between the two transcriptions as an
objective agreement signal that does not depend on the judge.

Usage: python scripts/judge_ocr.py --run Qwen3-VL-8B-Instruct [--judge-model Qwen/Qwen3-VL-4B-Instruct]
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.judge import judge_pair, split_by_tile  # noqa: E402
from ocr.pages import fit, load_page_image, split_rows  # noqa: E402
from ocr.text import cer  # noqa: E402
from ocr.vlm import VLM  # noqa: E402


MIN_TILE_CHARS = 150


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/vlm_ocr"))
    ap.add_argument("--judge-model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-side", type=int, default=2560)
    args = ap.parse_args()

    run_dir = args.out / args.run
    result_path = run_dir / "judge.jsonl"
    done = {json.loads(l)["page_id"] for l in result_path.open()} if result_path.exists() else set()
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((run_dir / "pages").glob("*.json"))]
    todo = [r for r in recs if r["page_id"] not in done]
    print(f"{len(recs)} transcribed pages, {len(todo)} to judge")

    vlm = VLM(args.judge_model, args.load_4bit) if todo else None
    with result_path.open("a", encoding="utf-8") as f:
        for r in todo:
            nzz = Path(r["pdf"]).with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
            tiles = split_rows(load_page_image(Path(r["pdf"])), r["tiles"])
            verdicts, cers = [], []
            nzz_tiles = split_by_tile(nzz, r["tile_texts"])
            for img, vlm_text, nzz_span in zip(tiles, r["tile_texts"], nzz_tiles):
                if len(vlm_text) < MIN_TILE_CHARS:  # masthead strips: scrambled in the NZZ OCR, too short to judge
                    verdicts.append("skip")
                    cers.append(None)
                    continue
                cers.append(cer(nzz_span, vlm_text))
                v = judge_pair(vlm, fit(img, args.max_side), nzz_span, vlm_text)
                verdicts.append({"x": "nzz", "y": "vlm", "tie": "tie"}[v])
            rec = {"page_id": r["page_id"], "year": r["year"], "verdicts": verdicts,
                   "cer_nzz_vs_vlm": cers}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(r["page_id"], verdicts, "tile CER(nzz,vlm)=", [None if c is None else round(c, 2) for c in cers], flush=True)

    total, by_decade = Counter(), defaultdict(Counter)
    for line in result_path.open():
        rec = json.loads(line)
        total.update(rec["verdicts"])
        by_decade[rec["year"] // 10 * 10].update(rec["verdicts"])
    print("\nTile verdicts:", dict(total))
    for d in sorted(by_decade):
        print(f"  {d}s: {dict(by_decade[d])}")


if __name__ == "__main__":
    main()
