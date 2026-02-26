from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingMetrics:
    ndcg_at_10: float
    precision_at_10: float
    ctr: float


@dataclass(frozen=True)
class ABTestSummary:
    delta_ctr: float
    ci_low: float
    ci_high: float
    p_value: float


def _dcg(scores: np.ndarray) -> float:
    return float(np.sum((2 ** scores - 1) / np.log2(np.arange(2, len(scores) + 2))))


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str = "label", k: int = 10) -> RankingMetrics:
    ndcg_values = []
    precision_values = []
    ctr_values = []

    for _, group in df.groupby("query_id"):
        ranked = group.sort_values(score_col, ascending=False)
        top_k = ranked.head(k)

        dcg = _dcg(top_k[label_col].to_numpy())
        ideal = _dcg(group.sort_values(label_col, ascending=False).head(k)[label_col].to_numpy())
        ndcg_values.append(dcg / ideal if ideal else 0.0)
        precision_values.append(float((top_k[label_col] > 0.55).mean()))
        ctr_values.append(float(top_k["clicked"].mean()))

    return RankingMetrics(
        ndcg_at_10=float(np.mean(ndcg_values)),
        precision_at_10=float(np.mean(precision_values)),
        ctr=float(np.mean(ctr_values)),
    )


def simulate_ab_test(
    baseline_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    k: int = 10,
    n_bootstrap: int = 300,
    seed: int = 7,
) -> ABTestSummary:
    baseline = (
        baseline_df.sort_values(["query_id", "semantic_similarity"], ascending=[True, False])
        .groupby("query_id")
        .head(k)
        .groupby("query_id")["clicked"]
        .mean()
    )
    treatment = (
        treatment_df.sort_values(["query_id", "prediction"], ascending=[True, False])
        .groupby("query_id")
        .head(k)
        .groupby("query_id")["clicked"]
        .mean()
    )

    aligned = pd.DataFrame({"b": baseline, "t": treatment}).dropna()
    deltas = aligned["t"] - aligned["b"]
    observed = float(deltas.mean())

    rng = np.random.default_rng(seed)
    boot = []
    arr = deltas.to_numpy()
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot.append(float(sample.mean()))

    ci_low, ci_high = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
    p_value = float((np.abs(boot) >= abs(observed)).mean())

    return ABTestSummary(delta_ctr=observed, ci_low=ci_low, ci_high=ci_high, p_value=p_value)
