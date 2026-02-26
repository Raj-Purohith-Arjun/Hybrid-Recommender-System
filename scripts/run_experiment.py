from __future__ import annotations

import argparse

from src.hybrid_rec.data import SyntheticConfig
from src.hybrid_rec.pipeline import HybridRecommender


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid recommendation experiment")
    parser.add_argument("--users", type=int, default=600)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--interactions", type=int, default=30)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    config = SyntheticConfig(
        n_users=args.users,
        n_items=args.items,
        interactions_per_user=args.interactions,
    )
    recommender = HybridRecommender()
    _, artifacts = recommender.run_experiment(config=config, top_n_retrieval=args.top_n)
    out = recommender.save_run(artifacts)

    print("Saved:", out)
    print("NDCG@10:", round(artifacts.test_metrics.ndcg_at_10, 4))
    print("Precision@10:", round(artifacts.test_metrics.precision_at_10, 4))
    print("CTR@10:", round(artifacts.test_metrics.ctr, 4))


if __name__ == "__main__":
    main()
