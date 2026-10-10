"""Taste filters: category, avoided cuisines, caffeine.

These run AFTER the safety gate and can only remove dishes, never add them, so they
cannot resurface anything the safety layer blocked. They express stated tastes
(`SoftPreferences`), not safety, so they live apart from `safety.py`.

Caffeine: a dish is dropped for a caffeine-free user when `Dish.caffeinated` is True or
any ingredient resolves to the CAFFEINE trait. Unknown caffeine content is NOT dropped
(this is a preference, not a health constraint).
"""
from __future__ import annotations

from typing import Optional, Sequence

from .knowledge_graph import KnowledgeGraph
from .schema import Dish, SoftPreferences, Trait

CATEGORIES = ("main", "dessert", "beverage")


def has_caffeine(dish: Dish, kg: KnowledgeGraph) -> bool:
    if dish.caffeinated is True:
        return True
    if dish.caffeinated is False:
        return False
    return any(Trait.CAFFEINE in kg.expand(m.text).traits for m in dish.ingredients)


def taste_filter(dishes: Sequence[Dish], soft: SoftPreferences, kg: KnowledgeGraph,
                 category: Optional[str] = "main") -> list[Dish]:
    """`category=None` keeps every category (the pre-category behaviour)."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; expected one of {CATEGORIES}")
    avoided = {c.lower() for c in soft.cuisines_avoided}
    out = []
    for d in dishes:
        if category is not None and d.category != category:
            continue
        if d.cuisine and d.cuisine.lower() in avoided:
            continue
        if not soft.caffeine_ok and d.category == "beverage" and has_caffeine(d, kg):
            continue
        out.append(d)
    return out
