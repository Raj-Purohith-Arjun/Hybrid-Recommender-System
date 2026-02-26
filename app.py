from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.hybrid_rec.data import SyntheticConfig
from src.hybrid_rec.pipeline import HybridRecommender


st.set_page_config(page_title="Hybrid Recommender Live Demo", layout="wide")
st.title("Hybrid Recommendation: Retrieval + Learning-to-Rank")
st.caption("Live experiment dashboard with offline metrics and A/B simulation")

with st.sidebar:
    st.header("Experiment controls")
    n_users = st.slider("Users", 200, 2000, 600, 50)
    n_items = st.slider("Items", 100, 2000, 400, 50)
    interactions_per_user = st.slider("Interactions/User", 10, 80, 30, 5)
    top_n_retrieval = st.slider("Stage-1 candidates/user", 5, 30, 15, 1)
    run = st.button("Run live experiment", type="primary")

if run:
    config = SyntheticConfig(
        n_users=n_users,
        n_items=n_items,
        interactions_per_user=interactions_per_user,
    )
    recommender = HybridRecommender()
    scored_df, artifacts = recommender.run_experiment(config=config, top_n_retrieval=top_n_retrieval)
    saved_path = recommender.save_run(artifacts)

    st.success(f"Run finished. Metrics saved to `{saved_path}`")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("NDCG@10", f"{artifacts.test_metrics.ndcg_at_10:.3f}", f"{artifacts.lift_ndcg_percent:.1f}%")
    k2.metric("Precision@10", f"{artifacts.test_metrics.precision_at_10:.3f}")
    k3.metric("CTR@10", f"{artifacts.test_metrics.ctr:.3f}", f"{artifacts.lift_ctr_percent:.1f}%")
    k4.metric("A/B ΔCTR", f"{artifacts.ab_test.delta_ctr:.4f}", f"p={artifacts.ab_test.p_value:.3f}")

    tab1, tab2, tab3 = st.tabs(["Overview", "A/B confidence", "Top results"])

    with tab1:
        split_df = (
            scored_df.groupby("split")[["prediction", "baseline_score", "label"]]
            .mean()
            .reset_index()
            .melt(id_vars="split", var_name="metric", value_name="value")
        )
        fig_bar = px.bar(split_df, x="metric", y="value", color="split", barmode="group")
        st.plotly_chart(fig_bar, use_container_width=True)

        test_df = scored_df[scored_df["split"] == "test"].copy()
        fig_scatter = px.scatter(
            test_df.sample(min(3000, len(test_df)), random_state=7),
            x="semantic_similarity",
            y="prediction",
            color="genre_match",
            opacity=0.6,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab2:
        ab_frame = pd.DataFrame(
            {
                "metric": ["delta_ctr", "ci_low", "ci_high", "p_value"],
                "value": [
                    artifacts.ab_test.delta_ctr,
                    artifacts.ab_test.ci_low,
                    artifacts.ab_test.ci_high,
                    artifacts.ab_test.p_value,
                ],
            }
        )
        st.dataframe(ab_frame, use_container_width=True)

    with tab3:
        test_df = scored_df[scored_df["split"] == "test"].copy()
        st.dataframe(
            test_df[
                [
                    "user_id",
                    "item_id",
                    "genre",
                    "favorite_genre",
                    "semantic_similarity",
                    "prediction",
                    "label",
                    "clicked",
                ]
            ]
            .sort_values("prediction", ascending=False)
            .head(50),
            use_container_width=True,
        )
else:
    st.info("Use the left sidebar to run a full live experiment.")
