"""Pairwise grading of two transcriptions (NZZ OCR vs. VLM) with a small VLM that sees the image.

A text-only judge could only rate how plausible the German looks; showing the scan lets it
check which candidate actually matches the print. Each comparison is run in both A/B orders
to cancel position bias; disagreeing orders count as a tie.
"""
import json
import re

from rapidfuzz import fuzz

from .text import normalize

JUDGE_PROMPT = (
    "The image is part of a historical German newspaper page (black-letter type). Below are two "
    "candidate transcriptions of its text. Decide which one is the more faithful transcription "
    "of what is printed in the image. Ignore line breaks and hyphenation; penalise wrong, "
    "missing, invented or repeated words and characters.\n\n"
    "=== Candidate A ===\n{a}\n\n=== Candidate B ===\n{b}\n\n"
    'Answer with JSON only: {{"winner": "A" | "B" | "tie"}}'
)


def split_by_tile(reference: str, tile_texts, min_score: float = 75.0) -> list:
    """Distribute the lines of the page-level ``reference`` over the VLM tiles.

    The NZZ OCR reads the page column by column, the VLM tile by tile, so the content of one
    tile is not contiguous in the reference. Each reference line goes to the tile that contains
    it best (ties and unclear short lines stay with the previous line's tile); lines that match
    no tile are dropped. Reference order is kept inside a tile.
    """
    tiles = [normalize(t) for t in tile_texts]
    out = [[] for _ in tiles]
    prev = 0
    for line in reference.splitlines():
        text = normalize(line)
        if len(text) < 3:
            continue
        scores = [fuzz.partial_ratio(text, t) for t in tiles]
        best = max(range(len(tiles)), key=scores.__getitem__)
        if scores[best] < min_score:
            continue
        if scores[prev] >= scores[best] - 1:  # (near-)tie: keep text contiguous
            best = prev
        out[best].append(line)
        prev = best
    return ["\n".join(lines) for lines in out]


def parse_verdict(answer: str) -> str:
    m = re.search(r"\{.*?\}", answer, re.S)
    try:
        w = json.loads(m.group(0))["winner"] if m else None
    except (json.JSONDecodeError, KeyError, TypeError):
        w = None
    return w if w in ("A", "B", "tie") else "tie"


def judge_pair(vlm, image, cand_x: str, cand_y: str) -> str:
    """Return 'x', 'y' or 'tie' for candidates x and y (both A/B orders must agree)."""
    first = parse_verdict(vlm.generate(JUDGE_PROMPT.format(a=cand_x, b=cand_y), [image], max_new_tokens=32))
    second = parse_verdict(vlm.generate(JUDGE_PROMPT.format(a=cand_y, b=cand_x), [image], max_new_tokens=32))
    first = {"A": "x", "B": "y", "tie": "tie"}[first]
    second = {"A": "y", "B": "x", "tie": "tie"}[second]
    return first if first == second else "tie"
