"""Check how faithfully a VLM run keeps the historical spelling of the Zenodo ground truth.

Reports, over the pages the run covers: line-CER, how many of the ground-truth words were
replaced by a near-identical word (edit distance <= 2, the typical signature of modernisation
such as bey->bei), the retention of old-spelling markers (y, th, c) and the most frequent
substitutions.

Usage: python scripts/check_modernization.py --run Qwen3-VL-8B-Instruct-page [--pages-of <other run>] [--top 25]
"""
import argparse
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein as L

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.pagexml import parse_page  # noqa: E402
from ocr.pages import SCANS_DIR, zenodo_pages  # noqa: E402
from ocr.text import line_cer  # noqa: E402


def light(s: str) -> str:
    s = unicodedata.normalize("NFC", s).replace("ſ", "s").replace("¬", "-")
    return " ".join(s.split())


def strip_punct(w: str) -> str:
    return w.strip(".,;:!?()\"'„“”-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--pages-of", help="restrict to the pages another run covers (fair comparison)")
    ap.add_argument("--out", type=Path, default=Path("data/vlm_ocr"))
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    ref_run = args.pages_of or args.run
    ids = {p.stem for p in (args.out / ref_run / "pages").glob("*.json")}
    gt_dir = next(Path("data/zenodo/extracted").glob("*/xml/NZZ_groundtruth"))
    n_words = n_sub = n_close = 0
    subs, cers = Counter(), []
    markers = {"y": [0, 0], "th": [0, 0], "ck": [0, 0]}  # [ground truth, hypothesis]
    for page, xml in zenodo_pages(gt_dir, SCANS_DIR):
        f = args.out / args.run / "pages" / f"{page.page_id}.txt"
        if page.page_id not in ids or not f.exists():
            continue
        raw = f.read_text(encoding="utf-8")
        hyp = light(raw)
        lines = [l.text for l in parse_page(xml)[1]]
        cers.append(line_cer(lines, raw))
        for m, c in markers.items():
            markers[m][0] += sum(len(re.findall(m, light(l).lower())) for l in lines)
            markers[m][1] += len(re.findall(m, hyp.lower()))
        for line in map(light, lines):
            if len(line) < 8:
                continue
            al = fuzz.partial_ratio_alignment(line, hyp)
            g, win = line.split(), hyp[al.dest_start:al.dest_end].split()
            n_words += len(g)
            for op in L.opcodes(g, win):
                if op.tag == "replace" and op.src_end - op.src_start == op.dest_end - op.dest_start:
                    for a, b in zip(g[op.src_start:op.src_end], win[op.dest_start:op.dest_end]):
                        a, b = strip_punct(a), strip_punct(b)
                        if a and a != b:
                            n_sub += 1
                            subs[(a, b)] += 1
                            n_close += L.distance(a, b) <= 2

    print(f"run {args.run}: {len(cers)} pages, line-CER {sum(cers) / max(len(cers), 1):.3f}")
    print(f"ground-truth words: {n_words}; substituted {n_sub} ({n_sub / max(n_words, 1):.1%}), "
          f"of which near-identical (modernisation-like) {n_close} ({n_close / max(n_words, 1):.1%})")
    print("old-spelling marker retention (hypothesis / ground truth): " +
          ", ".join(f"'{m}' {h}/{g} = {h / max(g, 1):.2f}" for m, (g, h) in markers.items()))
    print(f"top {args.top} substitutions (ground truth -> run):")
    for (a, b), v in subs.most_common(args.top):
        print(f"  {v:3d}  {a!r:20} -> {b!r}")


if __name__ == "__main__":
    main()
