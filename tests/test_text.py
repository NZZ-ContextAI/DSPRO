import pytest

from ocr.text import cer, evaluate_page, wer


def test_perfect_match():
    assert cer("Das ist gut", "Das ist gut") == 0.0
    assert wer("Das ist gut", "Das ist gut") == 0.0


def test_completely_wrong_is_non_negative_float():
    c, w = cer("abcdef", "xyzxyz"), wer("eins zwei drei", "vier fuenf sechs")
    assert isinstance(c, float) and isinstance(w, float)
    assert 0.0 <= c <= 1.0
    assert w >= 0.0


@pytest.mark.parametrize("fn", [cer, wer])
def test_empty_reference_returns_zero(fn):
    assert fn("", "anything") == 0.0
    assert fn("", "") == 0.0


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
