import importlib.util
from pathlib import Path

import pytest

missing = [name for name in ["numpy", "pandas", "sklearn"] if importlib.util.find_spec(name) is None]
if missing:
    pytest.skip(f"Missing optional dependencies: {', '.join(missing)}", allow_module_level=True)

from src.hybrid_rec.data import SyntheticConfig
from src.hybrid_rec.pipeline import HybridRecommender


def test_pipeline_runs_and_writes_metrics_file(tmp_path: Path):
    recommender = HybridRecommender(random_state=13)
    _, artifacts = recommender.run_experiment(SyntheticConfig(n_users=80, n_items=70, interactions_per_user=12, seed=13))

    assert 0.0 <= artifacts.test_metrics.ndcg_at_10 <= 1.0
    assert 0.0 <= artifacts.test_metrics.precision_at_10 <= 1.0
    assert 0.0 <= artifacts.test_metrics.ctr <= 1.0
    assert artifacts.ab_test.ci_low <= artifacts.ab_test.delta_ctr <= artifacts.ab_test.ci_high

    output = recommender.save_run(artifacts, output_dir=str(tmp_path))
    assert Path(output).exists()
