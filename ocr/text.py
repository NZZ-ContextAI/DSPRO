"""Character and word error rates for OCR evaluation, plus the text normalisation they can use.

Two ways to compare a hypothesis with a reference:

* ``cer``/``wer`` compare both texts in order. With ``norm=False`` (default) the comparison is
  diplomatic: every character counts. ``norm=True`` first applies ``normalize`` (NFC, long s -> s,
  line-end hyphenation removed, whitespace collapsed), so layout artefacts are not counted as errors.
* ``line_cer`` matches every reference line to its best window in the hypothesis. A different but
  valid reading order of columns is then not punished as if it were misread text.
"""
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

# line-end hyphenation: "widerle-\ngen" -> "widerlegen" (also "Dank—-\nbarkeit")
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])[-‐‑‒–—⸗¬]+[ \t]*\n[ \t]*")


def normalize(text: str) -> str:
    """NFC, long s -> s, dehyphenate line breaks, collapse whitespace."""
    text = unicodedata.normalize("NFC", text).replace("ſ", "s")
    text = _HYPHEN_BREAK.sub("", text)
    return " ".join(text.split())


def cer(reference: str, hypothesis: str, norm: bool = False) -> float:
    """Character Error Rate: edit distance / len(reference).

    An empty reference gives 0.0 for an empty hypothesis and 1.0 otherwise, so text a model
    invents on an empty region still counts as an error.
    """
    if norm:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return Levenshtein.distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str, norm: bool = False) -> float:
    """Word Error Rate: edit distance on whitespace tokens / number of reference words.

    Empty references are handled like in ``cer``.
    """
    if norm:
        reference, hypothesis = normalize(reference), normalize(hypothesis)
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return Levenshtein.distance(ref_words, hyp_words) / len(ref_words)


def line_cer(ref_lines, hypothesis: str, min_len: int = 8) -> float:
    """Reading-order-independent CER: each reference line is matched to its best window in ``hypothesis``.

    Lines shorter than ``min_len`` (after light normalisation) are skipped, they match almost
    anywhere. Hyphens are kept so hyphenated line ends still match.
    """
    from rapidfuzz import fuzz

    def light(s):
        # "¬" marks a line-end hyphen in some ground truths
        return " ".join(unicodedata.normalize("NFC", s).replace("ſ", "s").replace("¬", "-").split())

    hyp = light(hypothesis)
    dist = total = 0
    for line in map(light, ref_lines):
        if len(line) < min_len:
            continue
        al = fuzz.partial_ratio_alignment(line, hyp)
        dist += Levenshtein.distance(line, hyp[al.dest_start:al.dest_end]) if al else len(line)
        total += len(line)
    return dist / total if total else 0.0


def evaluate_page(reference: str, hypothesis: str, norm: bool = False) -> dict:
    """Return {"cer": float, "wer": float} for one page."""
    return {"cer": cer(reference, hypothesis, norm), "wer": wer(reference, hypothesis, norm)}


def score(reference: str, hypothesis: str) -> dict:
    """All rates used by the benchmark for one tile or page.

    ``cer``/``wer``: diplomatic, in order. ``cer_norm``: after ``normalize``. ``recall_cer``:
    missing or misread reference text, ``precision_cer``: invented or duplicated text (both
    ignore reading order).
    """
    return {
        "cer": cer(reference, hypothesis),
        "wer": wer(reference, hypothesis),
        "cer_norm": cer(reference, hypothesis, norm=True),
        "recall_cer": line_cer(reference.splitlines(), hypothesis, min_len=4),
        "precision_cer": line_cer(hypothesis.splitlines(), reference, min_len=4),
    }
