"""Hybrid recommender package."""

from .data import SyntheticConfig, build_synthetic_dataset
from .evaluate import ABTestSummary, RankingMetrics, evaluate_ranking, simulate_ab_test
from .pipeline import HybridRecommender
from .retrieval import retrieve_candidates

__all__ = [
    "ABTestSummary",
    "HybridRecommender",
    "RankingMetrics",
    "SyntheticConfig",
    "build_synthetic_dataset",
    "evaluate_ranking",
    "retrieve_candidates",
    "simulate_ab_test",
]
