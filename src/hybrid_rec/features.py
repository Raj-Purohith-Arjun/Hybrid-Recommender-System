from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEncoder:
    """Sentence embedding wrapper with TF-IDF fallback for local reproducibility."""

    def __init__(self, max_features: int = 128):
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    def fit(self, corpus: list[str]) -> "SemanticEncoder":
        self.vectorizer.fit(corpus)
        return self

    def similarity(self, left_text: list[str], right_text: list[str]) -> np.ndarray:
        left_vec = self.vectorizer.transform(left_text)
        right_vec = self.vectorizer.transform(right_text)
        similarities = np.array([
            cosine_similarity(left_vec[i], right_vec[i])[0, 0] for i in range(left_vec.shape[0])
        ])
        return similarities


def build_feature_table(users: pd.DataFrame, items: pd.DataFrame, interactions: pd.DataFrame) -> pd.DataFrame:
    merged = interactions.merge(users, on="user_id").merge(items, on="item_id")
    merged["genre_match"] = (merged["favorite_genre"] == merged["genre"]).astype(float)
    merged["price_bucket"] = pd.cut(merged["price"], bins=[0, 25, 100, 250, 1000], labels=[0, 1, 2, 3]).astype(int)
    merged["profile_text"] = merged["favorite_genre"] + " shopper " + merged["region"]

    encoder = SemanticEncoder().fit((items["text"].tolist() + merged["profile_text"].tolist()))
    merged["semantic_similarity"] = encoder.similarity(
        merged["text"].tolist(),
        merged["profile_text"].tolist(),
    )

    return merged
