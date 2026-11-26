import importlib.util
import json
from pathlib import Path

from PIL import Image

from ocr.engines import Engine

spec = importlib.util.spec_from_file_location(
    "ocr_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "ocr_benchmark.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

GOLD = {"1785-01-01_01-00001_t0": "Zürich, den 12. Jenner.\nAus Paris wird gemeldet,",
        "1795-03-03_01-00002_t1": "daß der König abgereiset sey."}


class FakeEngine(Engine):
    spec = "fake"

    def transcribe(self, image):
        return "Zürich, den 12. Jenner.\nAus Paris wird gemeldet,"


def make_exp(tmp_path: Path) -> Path:
    exp = tmp_path / "exp"
    (exp / "tiles").mkdir(parents=True)
    (exp / "gold").mkdir()
    pages = [{"page_id": "1785-01-01_01-00001", "year": 1785, "tiles": 2},
             {"page_id": "1795-03-03_01-00002", "year": 1795, "tiles": 2},
             {"page_id": "1800-05-05_01-00001", "year": 1800, "tiles": 2}]
    (exp / "pages.json").write_text(json.dumps(pages), encoding="utf-8")
    for p in pages:
        for k in range(2):
            Image.new("L", (40, 20), 255).save(exp / "tiles" / f"{p['page_id']}_t{k}.jpg")
    for tile, text in GOLD.items():
        (exp / "gold" / f"{tile}.txt").write_text(text, encoding="utf-8")
    (exp / "gold" / "1800-05-05_01-00001_t0.draft.txt").write_text("not gold", encoding="utf-8")
    return exp


def test_run_writes_transcripts_for_gold_pages_and_resumes(tmp_path, monkeypatch):
    exp = make_exp(tmp_path)
    monkeypatch.setattr(bench, "make_engine", lambda s: FakeEngine())
    bench.main(["run", "--exp", str(exp), "--engines", "fake", "--only-gold"])
    out = exp / "transcripts" / "fake"
    assert sorted(p.name for p in out.glob("*.json")) == ["1785-01-01_01-00001.json", "1795-03-03_01-00002.json"]
    rec = json.loads((out / "1785-01-01_01-00001.json").read_text(encoding="utf-8"))
    assert len(rec["tile_texts"]) == 2 and rec["model"] == "fake" and rec["year"] == 1785

    calls = []
    monkeypatch.setattr(FakeEngine, "transcribe", lambda self, img: calls.append(1) or "")
    bench.main(["run", "--exp", str(exp), "--engines", "fake", "--only-gold"])
    assert calls == []  # everything was already done


def test_score_compares_sources_on_common_gold_tiles(tmp_path, monkeypatch):
    exp = make_exp(tmp_path)
    monkeypatch.setattr(bench, "make_engine", lambda s: FakeEngine())
    bench.main(["run", "--exp", str(exp), "--engines", "fake"])
    # a second source (e.g. a VLM) that only covers the first page
    vlm = exp / "transcripts" / "vlm"
    vlm.mkdir()
    (vlm / "1785-01-01_01-00001.json").write_text(json.dumps(
        {"tile_texts": [GOLD["1785-01-01_01-00001_t0"], ""], "seconds": 3.0}), encoding="utf-8")

    rows = bench.score_rows(exp)
    assert {(r["source"], r["tile"]) for r in rows} == {
        ("fake", "1785-01-01_01-00001_t0"), ("fake", "1795-03-03_01-00002_t1"), ("vlm", "1785-01-01_01-00001_t0")}
    summary, by_decade = bench.summarize(rows)
    assert list(summary["tiles"]) == [1, 1]  # only the tile both sources have
    assert summary.loc["vlm", "cer"] == 0.0 and summary.loc["fake", "cer"] == 0.0
    assert list(by_decade.index) == [1780]

    bench.main(["score", "--exp", str(exp)])
    assert (exp / "benchmark" / "summary.md").exists() and (exp / "benchmark" / "scores.csv").exists()


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bench.bootstrap_ci([0.01, 0.02, 0.03, 0.10, 0.02])
    assert lo <= 0.036 <= hi
    assert bench.bootstrap_ci([0.05] * 10) == (0.05, 0.05)
