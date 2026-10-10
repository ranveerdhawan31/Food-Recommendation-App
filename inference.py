"""Ingredient inference from dish names and descriptions.

Rule-based baseline behind the `IngredientExtractor` interface. A real NER or
LLM extractor can replace it as long as it returns `IngredientMention`s with
calibrated confidences. Output is always source="nlp", so dishes built from it
are never marked `ingredients_complete`.

Negations ("dairy-free") are deliberately NOT honored: ignoring them can only
make the system more conservative, never less.
"""
from __future__ import annotations

import re
from typing import Protocol

from .knowledge_graph import KnowledgeGraph, normalize
from .schema import Dish, IngredientMention

# Implied ingredients for well-known dish names (lower confidence: inferred, not listed).
DISH_PRIORS: dict[str, list[str]] = {
    "butter chicken": ["butter", "cream", "chicken", "tomato"],
    "caesar salad": ["caesar dressing", "lettuce", "bread"],
    "pad thai": ["noodles", "peanut", "fish sauce", "egg", "shrimp"],
    "carbonara": ["pasta", "egg", "bacon", "parmesan"],
    "pesto pasta": ["pasta", "pesto"],
    "hummus": ["hummus"],
    "falafel": ["chickpea", "sesame"],
    "tiramisu": ["egg", "cream", "flour", "wine"],
    "biryani": ["rice", "yogurt", "onion"],
    "ramen": ["noodles", "soy sauce", "egg", "broth"],
}

DESCRIPTION_CONF = 0.95  # explicitly mentioned in the description text
NAME_PRIOR_CONF = 0.70  # implied by a known dish name


class IngredientExtractor(Protocol):
    def extract(self, name: str, description: str) -> list[IngredientMention]: ...


class RuleBasedExtractor:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self._names = kg.all_names()  # longest first, so "peanut butter" beats "butter"

    def _scan(self, text: str) -> list[str]:
        remaining = " " + normalize(text) + " "
        found: list[str] = []
        for name in self._names:
            pattern = r"(?<![a-z0-9])" + re.escape(name) + r"(?:es|s)?(?![a-z0-9])"
            if re.search(pattern, remaining):
                found.append(name)
                remaining = re.sub(pattern, " ", remaining)  # consume the span
        return found

    def extract(self, name: str, description: str) -> list[IngredientMention]:
        mentions: dict[str, float] = {}
        for ing in self._scan(f"{name}. {description}"):
            mentions[ing] = max(mentions.get(ing, 0.0), DESCRIPTION_CONF)
        lowered = normalize(name)
        for dish_name, implied in DISH_PRIORS.items():
            if dish_name in lowered:
                for ing in implied:
                    mentions[ing] = max(mentions.get(ing, 0.0), NAME_PRIOR_CONF)
        return [IngredientMention(i, c, source="nlp") for i, c in sorted(mentions.items())]


def enrich_dish(dish: Dish, extractor: IngredientExtractor) -> Dish:
    """Fill ingredients from text when the provider gave no structured list."""
    if dish.ingredients and dish.ingredients_complete:
        return dish
    inferred = extractor.extract(dish.name, dish.description)
    known = {m.text for m in dish.ingredients}
    dish.ingredients = list(dish.ingredients) + [m for m in inferred if m.text not in known]
    dish.ingredients_complete = False  # inference can never prove completeness
    return dish
