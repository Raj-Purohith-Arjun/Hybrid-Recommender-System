# Hybrid Recommendation & Learning-to-Rank System

A **live-demo-ready**, end-to-end recommender project built with free tooling:
- **Stage 1 retrieval**: semantic candidate generation
- **Stage 2 reranking**: gradient-boosted learning-to-rank model
- **Metrics**: NDCG@10, Precision@10, CTR@10
- **A/B simulation**: bootstrap confidence interval and p-value proxy
- **Live UI**: Streamlit dashboard with visualizations

## What makes this end-to-end

This is not just notebook code. It includes the same project primitives engineers use in real teams:
- modular `src/` package
- executable CLI script for repeatable runs
- Streamlit app for stakeholder demos
- Docker container for portable deployment
- CI workflow for automated test checks

## Architecture

```text
Synthetic/User behavior data
          │
          ▼
Feature generation (semantic + tabular)
          │
          ▼
Retriever (top-N candidates / user)
          │
          ▼
Reranker (HistGradientBoostingRegressor)
          │
          ├── Offline ranking metrics (NDCG/Precision/CTR)
          └── Online proxy (A/B bootstrap on CTR)
```

## Repository

```text
.
├── app.py
├── scripts/run_experiment.py
├── src/hybrid_rec/
│   ├── data.py
│   ├── features.py
│   ├── retrieval.py
│   ├── evaluate.py
│   └── pipeline.py
├── tests/
├── Dockerfile
├── Makefile
└── .github/workflows/ci.yml
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
make install
make test
```

## Run live demo

```bash
make demo
```

Open `http://localhost:8501` and click **Run live experiment**.

## Run pipeline without UI (CLI)

```bash
make run
# or
python scripts/run_experiment.py --users 800 --items 700 --interactions 35 --top-n 20
```

This writes `artifacts/metrics.json` for reproducible run tracking.

## Deploy for free

### Option 1: Streamlit Community Cloud
1. Push repo to GitHub.
2. Create app in Streamlit Community Cloud.
3. Entrypoint: `app.py`.

### Option 2: Container-based deployment

```bash
docker build -t hybrid-rec-demo .
docker run -p 8501:8501 hybrid-rec-demo
```

## Notes

- Current semantic encoder uses TF-IDF fallback so the project is fully runnable without paid inference APIs.
- To move to production, swap in Sentence-BERT embeddings and a true LambdaMART ranker (e.g., LightGBM).
