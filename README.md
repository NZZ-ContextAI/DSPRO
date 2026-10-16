# NZZ ContextAI

**Retrieval-Augmented Generation (RAG) system for semantic search over NZZ archive articles**

> DSPRO1 FS26 — Hochschule Luzern  
> Team: Raphael Wunderlin, Enrico Tesan, Florentin Genge  
> Coach: Dimitris Mousadakos

---

## Problem Statement

NZZ journalists and editors work with a growing archive of tens of thousands of articles. Finding relevant prior reporting requires manual keyword searches across disparate tools — a slow, context-blind process that fails whenever the query intent does not match the exact wording in the article (vocabulary mismatch). A journalist asking *"How did the SNB respond to the strong franc?"* will miss dozens of relevant articles that discuss the topic without using the word "respond".

**Goal:** Replace keyword search with a RAG pipeline that understands the *meaning* of a query — not just its tokens — so journalists can ask questions in natural language and receive source-backed answers within seconds, regardless of how the archive authors phrased the same topic.

**Target users:** NZZ journalists and editors who need to cross-reference the archive during story research.

---

## Screenshots

| Welcome Screen | Example Query & Answer | Sources |
|:-:|:-:|:-:|
| ![Welcome Screen](media/screenshot_welcome.png) | ![Answer View](media/screenshot_answer.png) | ![Source Cards](media/screenshot_sources.png) |

---

## Overview

NZZ ContextAI allows journalists and editors to ask questions against the NZZ archive and receive source-backed answers. The system combines semantic search (embedding + reranking with RRF fusion) with local LLM generation via Ollama. The interface supports German and English, streaming responses, conversation history, and configurable retrieval parameters at runtime.

**Pipeline:**
```
JSON articles → BeautifulSoup → Chunking → Qwen3-Embedding → ChromaDB
                                                                   ↓
             Gemma4 (Ollama) ← RRF Fusion ← Reranking ← Semantic Search
```

---

## Project Structure

```
nzz-contextai/
├── src/
│   ├── config.py          # Central configuration (models, paths, parameters)
│   ├── ingest.py          # Ingestion pipeline (load → preprocess → chunk → embed)
│   ├── preprocess.py      # HTML cleaning (BeautifulSoup), normalization
│   ├── chunking.py        # Article chunking
│   ├── embed.py           # Embedding generation, ChromaDB storage
│   ├── retrieval.py       # Semantic search + reranking + RRF fusion
│   ├── generate.py        # Answer generation with Gemma4 via Ollama
│   ├── evaluate.py        # Metrics: Faithfulness, ROUGE-L, Precision, Recall
│   ├── experiment.py      # Evaluation runner with MLflow logging
│   ├── setup_models.py    # Download required models (run once)
│   └── prompts/
│       └── system_prompt.md
├── api/
│   └── main.py            # FastAPI backend (streaming, runtime config via PATCH /config)
├── visualization/         # React + Vite frontend
│   └── src/
│       ├── components/    # App, SourceCard, Cite, CopyButton, LoadingDots, Icons
│       ├── api.js         # API client (streaming, config, topics)
│       ├── i18n.jsx       # DE / EN translations
│       └── theme.js       # Light / dark theme tokens
├── ocr/                   # DSPRO2: Fraktur OCR (preprocessing, baseline, fine-tuning)
├── layout/                # DSPRO2: column/line/article segmentation, reassembly
├── normalization/         # DSPRO2: historical spelling normalization
├── eval/                  # DSPRO2: CER/WER evaluation module
├── scripts/
│   └── build_expected_answers.py  # Generate reference answers for ground truth
├── notebooks/
│   ├── Pipeline.ipynb             # Run the full pipeline (Ingest → Eval → Query)
│   ├── Baseline Evaluation.ipynb  # Compare Random / BM25 / Dense / RRF strategies
│   ├── Explorer Notebook.ipynb    # Inspect raw data & debug retrieval
│   └── Data Exploration.ipynb     # Dataset statistics & charts
├── demo/
│   └── app.py                     # Legacy Streamlit chat interface
├── data/
│   ├── raw/                       # NZZ JSON source files (not in git)
│   ├── chroma/                    # ChromaDB vector store (not in git)
│   ├── zenodo/                    # Zenodo Fraktur ground truth (data not in git)
│   └── eval/
│       └── ground_truth.jsonl     # Evaluation queries with reference answers
├── media/                         # Screenshots for README
├── tests/
│   └── test_pipeline.py
├── main.py                        # CLI query entry point
├── requirements.txt
├── Makefile
└── .env.example
```

---

## Setup

### 1. Install Python dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 2. Install Ollama and download models
```bash
curl -fsSL https://ollama.com/install.sh | sh
make setup
```

### 3. Install frontend dependencies
```bash
cd visualization && npm install
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

---

## Running the App

### Full stack (API + frontend together)
```bash
make dev
```
- FastAPI backend → [http://localhost:8000](http://localhost:8000)
- Vite dev server → [http://localhost:5173](http://localhost:5173)

### Separate processes
```bash
make api        # FastAPI backend only
make frontend   # React frontend only
```

---

## Running the Pipeline

```bash
# Index all articles (run on first setup or after new data arrives)
make ingest

# Index a single month
.venv/bin/python src/ingest.py --month 2025_12 --reset

# Generate reference answers for evaluation (run once)
make ground-truth

# Run full evaluation with MLflow tracking
make experiment

# Open MLflow UI → http://localhost:5000
make mlflow
```

Or run everything from the notebook: `notebooks/Pipeline.ipynb`

---

## Configuration

All default parameters are defined in [`src/config.py`](src/config.py). Runtime parameters can be changed without restart via `PATCH /config` or through the Settings panel in the UI.

| Parameter | Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model |
| `RERANKER_MODEL` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker |
| `LLM_MODEL` | `gemma4` | Generation model (Ollama) |
| `CHUNK_SIZE` | `1000` | Words per chunk |
| `CHUNK_OVERLAP` | `250` | Overlap between chunks |
| `EVAL_TOP_K_RETRIEVAL` | `20` | Candidates retrieved from ChromaDB |
| `EVAL_TOP_K_FINAL` | `5` | Kept after reranking |

---

## Baseline & Evaluation

### Baseline comparison

`notebooks/Baseline Evaluation.ipynb` benchmarks four retrieval strategies on the same ground-truth queries to prove the production pipeline outperforms simple heuristics:

| Strategy | Description |
|---|---|
| Random | Uniformly sampled articles — absolute lower bound |
| BM25 Keyword | Classic term-frequency matching, no embeddings |
| Dense Semantic | Vector search only (Qwen3-Embedding-0.6B) |
| **Dense + RRF** | Dense + BM25 fused via Reciprocal Rank Fusion **(production)** |

### MLflow metrics

The following metrics are logged per run:

| Metric | Description |
|---|---|
| `hit_at_1/5` | Does the correct article appear in top-K results? |
| `mrr` | Mean Reciprocal Rank |
| `precision_at_5` | Fraction of relevant chunks in top-5 |
| `recall_at_5` | Coverage of relevant articles |
| `faithfulness` | How well the answer is grounded in retrieved sources |
---

## License

Internal academic project — HSLU FS26. Not for public redistribution.
