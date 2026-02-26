from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .data import SyntheticConfig, build_synthetic_dataset
from .evaluate import ABTestSummary, RankingMetrics, evaluate_ranking, simulate_ab_test
from .features import build_feature_table
from .retrieval import retrieve_candidates


FEATURE_COLUMNS = ["genre_match", "price", "price_bucket", "semantic_similarity", "rank_position"]


@dataclass
class TrainingArtifacts:
    train_metrics: RankingMetrics
    test_metrics: RankingMetrics
    lift_ndcg_percent: float
    lift_ctr_percent: float
    ab_test: ABTestSummary


class HybridRecommender:
    def __init__(self, random_state: int = 7):
        self.random_state = random_state
        self.model = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, random_state=random_state)

    def run_experiment(self, config: SyntheticConfig = SyntheticConfig(), top_n_retrieval: int = 15) -> tuple[pd.DataFrame, TrainingArtifacts]:
        users, items, interactions = build_synthetic_dataset(config)
        dataset = build_feature_table(users, items, interactions)

        retrieved = retrieve_candidates(dataset, top_n=top_n_retrieval)
        train_df, test_df = train_test_split(retrieved, test_size=0.25, random_state=self.random_state)

        self.model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
        train_df = train_df.assign(prediction=self.model.predict(train_df[FEATURE_COLUMNS]))
        test_df = test_df.assign(prediction=self.model.predict(test_df[FEATURE_COLUMNS]))

        test_baseline = evaluate_ranking(test_df, score_col="semantic_similarity")
        train_metrics = evaluate_ranking(train_df, score_col="prediction")
        test_metrics = evaluate_ranking(test_df, score_col="prediction")

        lift_ndcg = ((test_metrics.ndcg_at_10 - test_baseline.ndcg_at_10) / max(test_baseline.ndcg_at_10, 1e-8)) * 100
        lift_ctr = ((test_metrics.ctr - test_baseline.ctr) / max(test_baseline.ctr, 1e-8)) * 100
        ab_test = simulate_ab_test(test_df, test_df)

        artifacts = TrainingArtifacts(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            lift_ndcg_percent=lift_ndcg,
            lift_ctr_percent=lift_ctr,
            ab_test=ab_test,
        )

        combined = pd.concat([
            train_df.assign(split="train", baseline_score=train_df["semantic_similarity"]),
            test_df.assign(split="test", baseline_score=test_df["semantic_similarity"]),
        ])
        return combined, artifacts

    def save_run(self, artifacts: TrainingArtifacts, output_dir: str = "artifacts") -> str:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_metrics": asdict(artifacts.train_metrics),
            "test_metrics": asdict(artifacts.test_metrics),
            "lift_ndcg_percent": artifacts.lift_ndcg_percent,
            "lift_ctr_percent": artifacts.lift_ctr_percent,
            "ab_test": asdict(artifacts.ab_test),
        }
        target = out / "metrics.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(target)
