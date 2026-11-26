import pytest

from ocr.text import cer, evaluate_page, line_cer, normalize, score, wer


def test_perfect_match():
    assert cer("Das ist gut", "Das ist gut") == 0.0
    assert wer("Das ist gut", "Das ist gut") == 0.0


def test_completely_wrong_is_non_negative_float():
    c, w = cer("abcdef", "xyzxyz"), wer("eins zwei drei", "vier fuenf sechs")
    assert isinstance(c, float) and isinstance(w, float)
    assert 0.0 <= c <= 1.0
    assert w >= 0.0


@pytest.mark.parametrize("fn", [cer, wer])
def test_empty_reference(fn):
    assert fn("", "") == 0.0
    assert fn("", "anything") == 1.0  # invented text on an empty region is an error


def test_single_character_substitution():
    assert cer("Zeitung", "Zeitunq") == pytest.approx(1 / len("Zeitung"))


def test_one_word_substituted_in_five():
    ref = "die alte Zeitung wurde gedruckt"
    hyp = "die alte Zeitung wurde gedrukt"
    assert wer(ref, hyp) == pytest.approx(0.2)


def test_evaluate_page_keys():
    result = evaluate_page("ab cd", "ab cx")
    assert set(result) == {"cer", "wer"}
    assert result["wer"] == pytest.approx(0.5)


def test_raw_cer_counts_long_s_and_hyphenation_norm_does_not():
    assert cer("wider-\nlegen", "widerlegen") > 0
    assert cer("wider-\nlegen", "widerlegen", norm=True) == 0.0
    assert cer("ſeyn", "seyn", norm=True) == 0.0
    assert normalize("  Dank¬\nbarkeit  ſo ") == "Dankbarkeit so"


def test_line_cer_ignores_column_order():
    lines = ["Zürich, den 12. Jenner.", "Aus Paris wird gemeldet, daß"]
    swapped = "\n".join(reversed(lines))
    assert line_cer(lines, swapped) == 0.0
    assert 0 < line_cer(lines, swapped.replace("gemeldet", "gemeldct")) < 0.1


def test_score_keys():
    assert set(score("ab cd", "ab cx")) == {"cer", "wer", "cer_norm", "recall_cer", "precision_cer"}
