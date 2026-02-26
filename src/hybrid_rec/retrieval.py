from __future__ import annotations

import pandas as pd


def retrieve_candidates(interactions: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Stage-1 retriever: keep top semantic candidates per query/user."""
    ranked = interactions.sort_values(["query_id", "semantic_similarity"], ascending=[True, False])
    return ranked.groupby("query_id", group_keys=False).head(top_n).copy()
