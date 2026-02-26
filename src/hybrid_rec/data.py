from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_users: int = 500
    n_items: int = 300
    interactions_per_user: int = 30
    seed: int = 7


def build_synthetic_dataset(config: SyntheticConfig = SyntheticConfig()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate users, items, and interactions for ranking experiments."""
    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)

    genres = ["books", "sports", "fashion", "electronics", "gaming", "beauty"]

    users = pd.DataFrame(
        {
            "user_id": np.arange(config.n_users),
            "age": np_rng.integers(18, 65, size=config.n_users),
            "region": np_rng.choice(["NA", "EU", "APAC", "LATAM"], size=config.n_users),
            "favorite_genre": np_rng.choice(genres, size=config.n_users),
        }
    )

    items = pd.DataFrame(
        {
            "item_id": np.arange(config.n_items),
            "genre": np_rng.choice(genres, size=config.n_items),
            "price": np_rng.uniform(5, 400, size=config.n_items).round(2),
            "text": [
                f"{genre} product with quality score {np_rng.uniform(0, 1):.2f}"
                for genre in np_rng.choice(genres, size=config.n_items)
            ],
        }
    )

    rows: list[dict] = []
    for user in users.itertuples(index=False):
        sampled_items = rng.sample(range(config.n_items), config.interactions_per_user)
        for rank, item_id in enumerate(sampled_items, start=1):
            item = items.iloc[item_id]
            genre_match = float(user.favorite_genre == item.genre)
            price_affinity = max(0.0, 1 - (item.price / 450))
            noise = np_rng.normal(0, 0.15)
            relevance = min(1.0, max(0.0, 0.55 * genre_match + 0.35 * price_affinity + noise))

            rows.append(
                {
                    "user_id": user.user_id,
                    "item_id": item_id,
                    "query_id": user.user_id,
                    "rank_position": rank,
                    "label": relevance,
                    "clicked": int(relevance > 0.55),
                    "watch_time": round(relevance * np_rng.uniform(15, 90), 2),
                }
            )

    interactions = pd.DataFrame(rows)
    return users, items, interactions
