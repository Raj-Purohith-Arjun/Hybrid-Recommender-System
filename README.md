# 🎯 Hybrid Recommendation & Learning-to-Rank System

> A **live-demo-ready**, end-to-end recommender system — built entirely with free, open-source tooling. No paid APIs, no cloud credentials, no GPU required.

[![CI](https://github.com/Raj-Purohith-Arjun/Hybrid-Recommender-System/actions/workflows/ci.yml/badge.svg)](https://github.com/Raj-Purohith-Arjun/Hybrid-Recommender-System/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 Table of Contents

1. [What Is This?](#-what-is-this)
2. [What makes this end-to-end](#what-makes-this-end-to-end)
3. [System Architecture](#-system-architecture)
4. [Repository Layout](#-repository-layout)
5. [Module Reference](#-module-reference)
6. [Prerequisites](#-prerequisites)
7. [Quickstart](#-quickstart)
8. [Run live demo](#-run-live-demo)
9. [Run pipeline without UI (CLI)](#-run-pipeline-without-ui-cli)
10. [Configuration Options](#-configuration-options)
11. [Understanding the Metrics](#-understanding-the-metrics)
12. [A/B Testing Simulation](#-ab-testing-simulation)
13. [Deploy for free](#-deploy-for-free-using-below-options)
14. [Extending the System](#-extending-the-system)
15. [Running the Tests](#-running-the-tests)
16. [Contributing](#-contributing)

---

## 🤔 What Is This?

This project is a **two-stage hybrid recommender system** — the same design pattern used by Netflix, YouTube, and e-commerce platforms at scale:

| Stage | What it does | Implementation |
|-------|-------------|----------------|
| **Stage 1 — Retrieval** | Quickly narrows millions of items to a short candidate list per user using semantic similarity | TF-IDF cosine similarity (`SemanticEncoder`) |
| **Stage 2 — Reranking** | Scores and reorders those candidates using richer features and a trained model | `HistGradientBoostingRegressor` (sklearn) |

On top of the core pipeline, the project adds:

- 📊 **Offline ranking metrics** — NDCG@10, Precision@10, CTR@10
- 🔬 **A/B test simulation** — bootstrap confidence intervals and p-value proxy
- 🖥️ **Interactive Streamlit dashboard** — tune parameters and visualize results live
- 🐳 **Docker image** — one-command portable deployment
- ✅ **CI with pytest** — automated regression safety net

---

## What makes this end-to-end

This is not just notebook code. It includes the same project primitives engineers use in real teams:

- Modular `src/` package with clearly separated concerns
- Executable CLI script for repeatable, scriptable runs
- Streamlit app for stakeholder demos without any coding
- Docker container for portable, environment-agnostic deployment
- CI workflow for automated test checks on every push

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────┐
│        Synthetic Dataset Builder     │
│  users · items · interaction log     │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│         Feature Engineering          │
│  genre_match · price_bucket          │
│  semantic_similarity (TF-IDF cosine) │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│      Stage 1 · Semantic Retriever    │
│  Top-N candidates per user           │
│  ranked by semantic_similarity       │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────┐
│      Stage 2 · Learning-to-Rank      │
│  HistGradientBoostingRegressor       │
│  trained on relevance labels         │
└──────────┬─────────────┬─────────────┘
           │             │
           ▼             ▼
  Offline Metrics    A/B Simulation
  NDCG · Prec · CTR  Bootstrap CI · p-value
```

**Data flow in detail:**

1. `data.py` generates a fully synthetic dataset of users, items, and interactions with realistic relevance scores.
2. `features.py` merges the tables and computes tabular and semantic features.
3. `retrieval.py` keeps only the top-N candidates per user (Stage 1).
4. `pipeline.py` trains the reranker, splits train/test, and collects all artifacts.
5. `evaluate.py` computes NDCG, Precision, CTR, and runs the A/B bootstrap.
6. `app.py` ties everything together into the Streamlit UI.

---

## 📁 Repository Layout

```text
Hybrid-Recommender-System/
├── app.py                        # Streamlit dashboard entry-point
├── scripts/
│   └── run_experiment.py         # CLI runner (no UI required)
├── src/
│   └── hybrid_rec/
│       ├── __init__.py
│       ├── data.py               # Synthetic dataset generation
│       ├── features.py           # Feature engineering + SemanticEncoder
│       ├── retrieval.py          # Stage-1 candidate retriever
│       ├── evaluate.py           # Ranking metrics + A/B test simulation
│       └── pipeline.py           # End-to-end orchestration
├── tests/
│   ├── test_pipeline.py          # Integration test for the full pipeline
│   └── test_readme.py            # Smoke test that key README sections exist
├── artifacts/                    # Auto-created; stores metrics.json after runs
├── Dockerfile                    # Production-ready container image
├── Makefile                      # Convenience commands (install/test/demo/run)
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest configuration
└── .github/workflows/ci.yml      # GitHub Actions CI workflow
```

---

## 📦 Module Reference

### `src/hybrid_rec/data.py`

Generates a fully in-memory, reproducible synthetic dataset.

```python
from src.hybrid_rec.data import SyntheticConfig, build_synthetic_dataset

config = SyntheticConfig(n_users=500, n_items=300, interactions_per_user=30, seed=7)
users, items, interactions = build_synthetic_dataset(config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_users` | `500` | Number of synthetic users |
| `n_items` | `300` | Number of synthetic items |
| `interactions_per_user` | `30` | Interactions logged per user |
| `seed` | `7` | Random seed for full reproducibility |

---

### `src/hybrid_rec/features.py`

Merges user/item/interaction tables and computes features:

| Feature | Type | Description |
|---------|------|-------------|
| `genre_match` | float (0/1) | Whether the item genre matches the user's favourite genre |
| `price_bucket` | int (0–3) | Discretised price band |
| `semantic_similarity` | float | TF-IDF cosine similarity between item text and user profile text |

---

### `src/hybrid_rec/retrieval.py`

Stage-1 retriever: keeps the top-N candidates per user sorted by `semantic_similarity`.

```python
from src.hybrid_rec.retrieval import retrieve_candidates

candidates = retrieve_candidates(feature_table, top_n=15)
```

---

### `src/hybrid_rec/evaluate.py`

Computes offline ranking metrics and runs the A/B bootstrap simulation.

**Metrics returned by `evaluate_ranking()`:**

| Metric | Formula / Description |
|--------|-----------------------|
| `ndcg_at_10` | Normalised Discounted Cumulative Gain at cutoff 10 |
| `precision_at_10` | Fraction of top-10 items with relevance > 0.55 |
| `ctr` | Mean click rate in the top-10 |

**A/B test returned by `simulate_ab_test()`:**

| Field | Description |
|-------|-------------|
| `delta_ctr` | Observed CTR lift (treatment − baseline) |
| `ci_low / ci_high` | 95 % bootstrap confidence interval |
| `p_value` | Fraction of bootstrap samples exceeding the observed delta |

---

### `src/hybrid_rec/pipeline.py`

Orchestrates the full experiment in a single call:

```python
from src.hybrid_rec.pipeline import HybridRecommender
from src.hybrid_rec.data import SyntheticConfig

rec = HybridRecommender(random_state=7)
scored_df, artifacts = rec.run_experiment(
    config=SyntheticConfig(n_users=600, n_items=400),
    top_n_retrieval=15,
)
rec.save_run(artifacts)  # writes artifacts/metrics.json
```

---

## 🛠️ Prerequisites

- **Python 3.11+**
- **pip** (comes with Python)
- **Docker** (optional — only for container deployment)
- **Git**

All Python dependencies are pinned in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical operations |
| `pandas` | Tabular data manipulation |
| `scikit-learn` | TF-IDF encoder + gradient-boosting reranker |
| `plotly` | Interactive charts in the dashboard |
| `streamlit` | Web UI framework |
| `pytest` | Test runner |

---

## ⚡ Quickstart

```bash
# 1. Clone the repository
git clone https://github.com/Raj-Purohith-Arjun/Hybrid-Recommender-System.git
cd Hybrid-Recommender-System

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
make install                     # equivalent to: pip install -r requirements.txt

# 4. Run the test suite to verify everything works
make test
```

Expected output for `make test`:
```
..                                         [100%]
2 passed in X.XXs
```

---

## Run live demo

```bash
make demo
```

This launches the Streamlit dashboard at **http://localhost:8501**.

From the sidebar you can control:

| Control | Range | Description |
|---------|-------|-------------|
| **Users** | 200 – 2 000 | How many synthetic users to generate |
| **Items** | 100 – 2 000 | How many synthetic items to generate |
| **Interactions/User** | 10 – 80 | Interaction density per user |
| **Stage-1 candidates/user** | 5 – 30 | Top-N items passed to the reranker |

Click **Run live experiment** and explore three tabs:

- **Overview** — bar chart of mean scores (train vs. test) and a scatter of semantic similarity vs. predicted relevance
- **A/B confidence** — table of bootstrap A/B statistics
- **Top results** — sortable table of the 50 highest-ranked test predictions

---

## 💻 Run pipeline without UI (CLI)

```bash
make run
# or with custom parameters:
python scripts/run_experiment.py --users 800 --items 700 --interactions 35 --top-n 20
```

**All CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--users` | `600` | Number of synthetic users |
| `--items` | `400` | Number of synthetic items |
| `--interactions` | `30` | Interactions per user |
| `--top-n` | `15` | Stage-1 retrieval cutoff |

The script prints metrics to stdout and writes `artifacts/metrics.json`:

```json
{
  "train_metrics": { "ndcg_at_10": 0.832, "precision_at_10": 0.741, "ctr": 0.523 },
  "test_metrics":  { "ndcg_at_10": 0.801, "precision_at_10": 0.712, "ctr": 0.498 },
  "lift_ndcg_percent": 12.4,
  "lift_ctr_percent": 9.7,
  "ab_test": { "delta_ctr": 0.012, "ci_low": -0.005, "ci_high": 0.028, "p_value": 0.14 }
}
```

---

## ⚙️ Configuration Options

The central configuration object is `SyntheticConfig` in `data.py`:

```python
from src.hybrid_rec.data import SyntheticConfig

config = SyntheticConfig(
    n_users=1000,           # scale up user base
    n_items=500,            # scale up item catalogue
    interactions_per_user=40,
    seed=42,                # change for a different random draw
)
```

The reranker hyper-parameters live in `pipeline.py`:

```python
self.model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.06,
    random_state=random_state,
)
```

---

## 📐 Understanding the Metrics

### NDCG@10 (Normalised Discounted Cumulative Gain)

Measures ranking quality. Higher is better (max = 1.0). Items ranked at the top contribute more than items at the bottom, weighted by `log2(rank + 1)`.

```
DCG  = Σ (2^relevance - 1) / log2(rank + 1)   for top-10
NDCG = DCG / ideal_DCG
```

### Precision@10

The fraction of the top-10 recommended items that are "relevant" (relevance > 0.55).

### CTR@10 (Click-Through Rate)

The mean click indicator across the top-10 recommendations. In this synthetic setup, an item is considered clicked when its relevance exceeds 0.55.

### Lift %

The percentage improvement of the reranker over the baseline (semantic similarity alone):

```
lift_ndcg = (ndcg_reranker - ndcg_baseline) / ndcg_baseline × 100
```

---

## 🔬 A/B Testing Simulation

The A/B test compares two ranking policies for the same users:

- **Baseline** — items ranked by `semantic_similarity` (Stage 1 only)
- **Treatment** — items ranked by the reranker `prediction` (Stage 1 + Stage 2)

The bootstrap procedure (`n_bootstrap=300`) repeatedly resamples the per-user CTR deltas to build an empirical null distribution, then reports:

- **`delta_ctr`** — the raw observed difference
- **95 % CI** — `[ci_low, ci_high]`
- **`p_value`** — an approximate two-sided p-value

> ⚠️ This is a proxy simulation (both arms use the same underlying data). For a real A/B test, route live traffic to each policy independently. Using the same data for both arms will overestimate statistical significance.

---

## Deploy for free using below options

### Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, choose this repository, set the branch and entrypoint to `app.py`.
4. Click **Deploy** — Streamlit handles hosting, no server required.

### Container-based deployment

Build and run the Docker image locally:

```bash
docker build -t hybrid-rec-demo .
docker run -p 8501:8501 hybrid-rec-demo
```

Then open **http://localhost:8501**.

The `Dockerfile` is based on `python:3.11-slim`, copies `requirements.txt` first to leverage layer caching, then copies the full source. The container exposes port `8501` and runs Streamlit with `--server.address=0.0.0.0` for external reachability.

---

## 🔧 Extending the System

### Swap in real embeddings (Sentence-BERT)

Replace `SemanticEncoder` in `features.py`:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

item_vecs = model.encode(items["text"].tolist(), normalize_embeddings=True)
user_vecs = model.encode(merged["profile_text"].tolist(), normalize_embeddings=True)
merged["semantic_similarity"] = (item_vecs * user_vecs).sum(axis=1)
```

### Use LightGBM LambdaMART for true LTR

```python
import lightgbm as lgb

train_data = lgb.Dataset(
    train_df[FEATURE_COLUMNS], label=train_df["label"],
    group=train_df.groupby("query_id").size().tolist(),
)
params = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10]}
model = lgb.train(params, train_data, num_boost_round=200)
```

### Add new features

Add columns in `build_feature_table()` (`features.py`) and include them in `FEATURE_COLUMNS` (`pipeline.py`):

```python
# Example: normalise price to a 0-1 affinity score
merged["price_affinity"] = (1 - merged["price"] / merged["price"].max()).clip(0, 1)
```

### Plug in a real dataset

Replace `build_synthetic_dataset()` with a loader that returns three DataFrames with the same schema (`user_id`, `item_id`, `query_id`, `label`, `clicked`).

---

## 🧪 Running the Tests

```bash
make test
# or directly:
pytest -q
```

| Test file | What it checks |
|-----------|---------------|
| `tests/test_pipeline.py` | Full pipeline runs end-to-end; metrics are in valid ranges; `metrics.json` is written |
| `tests/test_readme.py` | Key README sections exist (smoke-test for documentation completeness) |

To run a single test file:

```bash
pytest tests/test_pipeline.py -v
```

---

## 🤝 Contributing

Contributions are welcome! Here's the recommended workflow:

1. **Fork** the repository and create a new branch:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. **Install** dependencies and verify tests pass before making changes:
   ```bash
   make install && make test
   ```
3. **Make** your changes, add or update tests where appropriate.
4. **Run** the test suite and the live demo to confirm nothing is broken.
5. **Open a Pull Request** — describe what you changed and why.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> **Note:** The semantic encoder uses TF-IDF as a local fallback, so the entire project runs without any paid inference API. To move toward production quality, swap in Sentence-BERT embeddings and a LambdaMART ranker as described in the [Extending the System](#-extending-the-system) section.
