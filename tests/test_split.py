from collections import Counter

from ocr.split import make_split, page_year

PAGES = [f"nzz_{y}0101_0_0_a1_p1_1" for y in range(1780, 1948)]


def test_every_page_assigned_once():
    s = make_split(PAGES)
    assert set(s) == set(PAGES) and len(s) == len(PAGES)
    assert set(s.values()) == {"train", "val", "test"}


def test_deterministic_and_seed_dependent():
    assert make_split(PAGES) == make_split(PAGES)
    assert make_split(PAGES) != make_split(PAGES, seed=1)


def test_every_decade_in_every_split():
    s = make_split(PAGES)
    seen = {(page_year(p) // 10 * 10, v) for p, v in s.items()}
    for d in range(1780, 1948, 10):
        assert {(d, "train"), (d, "val"), (d, "test")} <= seen


def test_roughly_70_15_15():
    c = Counter(make_split(PAGES).values())
    assert 0.6 < c["train"] / len(PAGES) < 0.8
