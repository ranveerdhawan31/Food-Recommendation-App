"""Single-user recommendation pipeline.

Order is fixed: safety filter -> taste filter -> retrieval -> ranking -> exploration -> MMR.
The ranker only ever sees dishes that already passed `safe_catalog`, so no
score (or exploration noise) can resurface a blocked dish. The taste filter
(category, avoided cuisines, caffeine) only removes dishes.

`recommend()` serves one category at a time; `recommend_meal()` returns mains,
desserts and beverages together.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from .knowledge_graph import KnowledgeGraph
from .ranking import HeuristicRanker, Ranker, build_features, explore, mmr
from .retrieval import eta_minutes, retrieve
from .safety import SafetyResult, safe_catalog, verification_warning
from .schema import Context, Dish, Restaurant, UserProfile
from .taste import taste_filter


@dataclass
class Recommendation:
    dish: Dish
    restaurant: Restaurant
    score: float
    safety: SafetyResult
    eta_min: float
    warning: Optional[str] = None  # "Verify with the restaurant..." text, None when nothing to verify


def recommend(
    user: UserProfile,
    ctx: Context,
    dishes: Sequence[Dish],
    restaurants: Mapping[str, Restaurant],
    kg: KnowledgeGraph,
    *,
    ranker: Optional[Ranker] = None,
    impressions: Optional[Mapping[str, int]] = None,
    incident_allergens: Optional[Mapping[str, frozenset]] = None,
    n: int = 10,
    k_retrieve: int = 200,
    rng: Optional[random.Random] = None,
    explore_sigma: float = 0.3,
    category: Optional[str] = "main",
) -> list[Recommendation]:
    ranker = ranker or HeuristicRanker()
    rng = rng or random.Random()

    # 1. safety (hard gate, rules only)
    safety = {}
    safe_dishes = []
    for dish, res in safe_catalog(dishes, restaurants, user.constraints, kg, incident_allergens=incident_allergens):
        safety[dish.id] = res
        safe_dishes.append(dish)

    # 2. taste filter (stated tastes; can only remove dishes), then retrieval
    safe_dishes = taste_filter(safe_dishes, user.soft, kg, category)
    retrieved = retrieve(safe_dishes, restaurants, user.soft, ctx, k=k_retrieve)
    if not retrieved:
        return []

    # 3. ranking
    rows = [build_features(d, restaurants[d.restaurant_id], user.soft, ctx, sim) for d, sim in retrieved]
    scores = ranker.score(rows)

    # 4. exploration, 5. diversity
    imp = impressions or {}
    noisy = explore(scores, [imp.get(d.id, 0) for d, _ in retrieved], rng, explore_sigma)
    order = mmr([d for d, _ in retrieved], noisy, n)

    return [
        Recommendation(retrieved[i][0], restaurants[retrieved[i][0].restaurant_id], scores[i],
                       safety[retrieved[i][0].id], eta_minutes(restaurants[retrieved[i][0].restaurant_id], ctx),
                       verification_warning(safety[retrieved[i][0].id], user.constraints))
        for i in order
    ]


@dataclass
class MealSuggestion:
    mains: list
    desserts: list
    beverages: list


def recommend_meal(
    user: UserProfile,
    ctx: Context,
    dishes: Sequence[Dish],
    restaurants: Mapping[str, Restaurant],
    kg: KnowledgeGraph,
    *,
    n_main: int = 5,
    n_dessert: int = 3,
    n_beverage: int = 3,
    rng: Optional[random.Random] = None,
    **kwargs,
) -> MealSuggestion:
    """Mains, desserts and non-alcoholic beverages in one call.

    Desserts are returned only when the user wants sweets: an active craving or a
    standing `sweet_tooth` of at least 0.3. Pass `n_*=0` to skip a category.
    """
    rng = rng or random.Random()
    wants_sweets = ctx.sweet_craving > 0.0 or user.soft.sweet_tooth >= 0.3
    plan = [("main", n_main), ("dessert", n_dessert if wants_sweets else 0), ("beverage", n_beverage)]
    out = {c: recommend(user, ctx, dishes, restaurants, kg, n=n, category=c, rng=rng, **kwargs) if n else []
           for c, n in plan}
    return MealSuggestion(out["main"], out["dessert"], out["beverage"])
