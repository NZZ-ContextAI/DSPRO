"""Text normalisation and error rates for comparing transcriptions."""
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# line-end hyphenation: "widerle-\ngen" -> "widerlegen" (also "Dank—-\nbarkeit")
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])[-‐‑‒–—⸗]+[ \t]*\n[ \t]*")


def normalize(text: str) -> str:
    """NFC, long s -> s, dehyphenate line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).replace("ſ", "s")
    text = _HYPHEN_BREAK.sub("", text)
    return " ".join(text.split())


def cer(ref: str, hyp: str, norm: bool = True) -> float:
    """Character error rate of ``hyp`` against ``ref`` (0 = identical)."""
    if norm:
        ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def wer(ref: str, hyp: str, norm: bool = True) -> float:
    """Word error rate; words are mapped to ids so Levenshtein runs on sequences."""
    if norm:
        ref, hyp = normalize(ref), normalize(hyp)
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    ids = {w: i for i, w in enumerate(set(r) | set(h))}
    return Levenshtein.distance([ids[w] for w in r], [ids[w] for w in h]) / len(r)


def line_cer(gt_lines, hyp: str, min_len: int = 8) -> float:
    """Reading-order-independent CER: each ground-truth line is matched to its best window in ``hyp``.

    Page-level CER punishes a different (but valid) column order as if it were text errors;
    matching line by line avoids that. Hyphens are kept so hyphenated line ends still match.
    """
    from rapidfuzz import fuzz

    def light(s):
        # "¬" marks a line-end hyphen
        return " ".join(unicodedata.normalize("NFC", s).replace("\u017f", "s").replace("\u00ac", "-").split())

    hyp = light(hyp)
    dist = total = 0
    for line in map(light, gt_lines):
        if len(line) < min_len:
            continue
        al = fuzz.partial_ratio_alignment(line, hyp)
        dist += Levenshtein.distance(line, hyp[al.dest_start:al.dest_end]) if al else len(line)
        total += len(line)
    return dist / total if total else 0.0
