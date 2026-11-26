"""Character and word error rate metrics for OCR evaluation."""

from rapidfuzz.distance import Levenshtein


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate: edit distance / len(reference). 0.0 if reference is empty."""
    if not reference:
        return 0.0
    return Levenshtein.distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate: edit distance on whitespace tokens / len(reference.split()).

    Returns 0.0 if the reference has no words.
    """
    ref_words = reference.split()
    if not ref_words:
        return 0.0
    return Levenshtein.distance(ref_words, hypothesis.split()) / len(ref_words)


def evaluate_page(reference: str, hypothesis: str) -> dict:
    """Return {"cer": float, "wer": float} for one page."""
    return {"cer": cer(reference, hypothesis), "wer": wer(reference, hypothesis)}
