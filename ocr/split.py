"""Page-level train/val/test split, stratified by decade.

Pages (not lines) are split so that no page contributes lines to two splits.
Within each decade pages are shuffled with a fixed seed and assigned to
test, val and train, so every decade appears in every split.
"""
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

SEED = 42
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def page_year(page_id: str) -> int:
    return int(re.search(r"nzz_(\d{4})\d{4}", page_id).group(1))


def make_split(page_ids, seed=SEED, val_frac=VAL_FRAC, test_frac=TEST_FRAC) -> dict:
    """Return {page_id: 'train'|'val'|'test'}."""
    by_decade = defaultdict(list)
    for p in sorted(page_ids):
        by_decade[page_year(p) // 10 * 10].append(p)

    rng = random.Random(seed)
    out = {}
    for decade in sorted(by_decade):
        pages = by_decade[decade]
        rng.shuffle(pages)
        n = len(pages)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        for i, p in enumerate(pages):
            out[p] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def write_split(split: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["page_id", "year", "decade", "split"])
        for p in sorted(split):
            y = page_year(p)
            w.writerow([p, y, y // 10 * 10, split[p]])


def read_split(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        return {r["page_id"]: r["split"] for r in csv.DictReader(f)}


def decade_table(split: dict) -> str:
    """Markdown table: pages per decade and split."""
    counts = defaultdict(lambda: defaultdict(int))
    for p, s in split.items():
        counts[page_year(p) // 10 * 10][s] += 1
    rows = ["| Decade | Train | Val | Test | Total |", "|---|---|---|---|---|"]
    tot = defaultdict(int)
    for d in sorted(counts):
        c = counts[d]
        rows.append(f"| {d}s | {c['train']} | {c['val']} | {c['test']} | {sum(c.values())} |")
        for s in c:
            tot[s] += c[s]
    rows.append(f"| **Total** | {tot['train']} | {tot['val']} | {tot['test']} | {sum(tot.values())} |")
    return "\n".join(rows)
