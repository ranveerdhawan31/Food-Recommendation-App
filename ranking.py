"""Ranking.

Features describe taste and context ONLY. Allergies, diet and Halal are never
inputs: they were already enforced by the safety filter, and keeping them out
of the model means a probabilistic score can never trade safety for relevance.
`assert_no_safety_features` guards this (and is unit-tested).
"""
from __future__ import annotations

import math
import random
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

from .retrieval import cosine, embed_dish, eta_minutes
from .schema import Context, Dish, Restaurant, SoftPreferences

FEATURE_NAMES = [
    "similarity", "cuisine_affinity", "spice_gap", "price_ratio", "eta_ratio",
    "hunger_portion_fit", "cold_hot_fit", "rain_comfort_fit", "sweetness_fit", "tag_affinity",
]
_FORBIDDEN = ("allerg", "halal", "diet", "vegan", "vegetarian", "severity", "gluten", "pork", "alcohol")


def assert_no_safety_features(names: Sequence[str]) -> None:
    bad = [n for n in names if any(f in n.lower() for f in _FORBIDDEN)]
    if bad:
        raise ValueError(f"safety attributes must not be ML features: {bad}")


assert_no_safety_features(FEATURE_NAMES)


def target_sweetness(soft: SoftPreferences, ctx: Context) -> float:
    """0..1. A stated craving dominates the standing sweet tooth."""
    base = min(1.0, max(0.0, soft.sweet_tooth))
    craving = min(1.0, max(0.0, ctx.sweet_craving))
    return base if craving == 0.0 else 0.3 * base + 0.7 * craving


def _tag_affinity(dish: Dish, soft: SoftPreferences) -> float:
    prefs = {"dessert": soft.dessert_tags, "beverage": soft.beverage_tags}.get(dish.category, {})
    vals = [prefs.get(t, 0.5) - 0.5 for t in dish.tags]
    return sum(vals) / len(vals) if vals else 0.0  # -0.5 .. 0.5, 0 when unknown


def build_features(dish: Dish, restaurant: Restaurant, soft: SoftPreferences, ctx: Context,
                   similarity: float) -> dict[str, float]:
    target_portion = 0.5 + ctx.hunger  # 0.5 (snack) .. 1.5 (starving)
    cold, hot_day = ctx.temperature_c < 10, ctx.temperature_c > 28
    if cold:
        cold_hot = 1.0 if dish.is_hot else -0.5
    elif hot_day:
        cold_hot = -0.5 if dish.is_hot else 1.0
    else:
        cold_hot = 0.0
    return {
        "similarity": similarity,
        "cuisine_affinity": soft.cuisines.get(dish.cuisine, 0.5),
        "spice_gap": abs(dish.spice_level - soft.spice_tolerance) / 5.0,
        "price_ratio": dish.price / ctx.budget if ctx.budget else 1.0,
        "eta_ratio": eta_minutes(restaurant, ctx) / ctx.max_eta_min,
        "hunger_portion_fit": 1.0 - min(1.0, abs(dish.portion - target_portion)),
        "cold_hot_fit": cold_hot,
        "rain_comfort_fit": 0.5 if (ctx.raining and dish.is_hot) else 0.0,
        # mains are neutral on sweetness; it only matters for desserts and drinks
        "sweetness_fit": (1.0 - abs(dish.sweetness / 5.0 - target_sweetness(soft, ctx)))
        if dish.category != "main" else 0.0,
        "tag_affinity": _tag_affinity(dish, soft),
    }


class Ranker(Protocol):
    def score(self, rows: Sequence[dict[str, float]]) -> list[float]: ...


class HeuristicRanker:
    """Transparent linear baseline; used until enough feedback exists to train a GBDT."""

    def __init__(self, weights: Optional[dict[str, float]] = None) -> None:
        self.w = weights or {
            "similarity": 1.0, "cuisine_affinity": 1.5, "spice_gap": -1.0, "price_ratio": -0.5,
            "eta_ratio": -0.7, "hunger_portion_fit": 0.8, "cold_hot_fit": 0.6, "rain_comfort_fit": 0.3,
            "sweetness_fit": 1.2, "tag_affinity": 1.0,
        }
        assert_no_safety_features(self.w)

    def score(self, rows):
        return [sum(self.w.get(k, 0.0) * v for k, v in r.items()) for r in rows]


class XGBRanker:
    """Adapter for a trained xgboost model (not exercised in the unit tests)."""

    def __init__(self, booster) -> None:
        self.booster = booster

    def score(self, rows):
        import xgboost as xgb  # lazy: optional dependency

        mat = np.array([[r[n] for n in FEATURE_NAMES] for r in rows])
        return self.booster.predict(xgb.DMatrix(mat, feature_names=FEATURE_NAMES)).tolist()


def explore(scores: Sequence[float], impressions: Sequence[int], rng: random.Random, sigma: float = 0.3) -> list[float]:
    """Thompson-style noise that shrinks as an item accumulates impressions."""
    return [s + rng.gauss(0.0, sigma / math.sqrt(1 + n)) for s, n in zip(scores, impressions)]


def dish_similarity(a: Dish, b: Dish) -> float:
    s = 0.5 * (a.cuisine == b.cuisine) + 0.3 * (a.restaurant_id == b.restaurant_id)
    return s + 0.2 * max(0.0, cosine(embed_dish(a), embed_dish(b)))


def mmr(items: Sequence[Dish], relevance: Sequence[float], k: int, lam: float = 0.7,
        sim: Callable[[Dish, Dish], float] = dish_similarity) -> list[int]:
    """Maximal Marginal Relevance. Returns indices of the chosen items in order."""
    if not items:
        return []
    lo, hi = min(relevance), max(relevance)
    rel = [(r - lo) / (hi - lo) if hi > lo else 1.0 for r in relevance]
    chosen: list[int] = []
    pool = list(range(len(items)))
    while pool and len(chosen) < k:
        def gain(i: int) -> float:
            penalty = max((sim(items[i], items[j]) for j in chosen), default=0.0)
            return lam * rel[i] - (1 - lam) * penalty
        best = max(pool, key=lambda i: (gain(i), -i))
        chosen.append(best)
        pool.remove(best)
    return chosen
