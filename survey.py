"""Interactive survey (cold start).

* Hard constraints are asked explicitly, always first, and are built only from
  the user's answers by `build_constraints` (never inferred).
* Cuisines the user states outright (liked / avoided) are applied with `SoftSurvey.declare`
  and never re-asked. Sweet, dessert and beverage tastes come from `build_taste_preferences`.
  Alcoholic drinks are never offered (see safety.py), so there is no alcohol question here.
* Soft cuisine tastes use a Beta belief per cuisine. For a noiseless yes/no
  answer the expected information gain equals the Bernoulli entropy of the
  current belief, so the next question is the most uncertain cuisine.
  Stops after `max_questions` or when every belief is confident.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .schema import Allergen, Diet, HalalLevel, Severity, SoftPreferences, UserConstraints

HARD_QUESTIONS = [
    ("allergies", "Do you have any food allergies?", [a.value for a in Allergen]),
    ("diet", "Do you follow a dietary pattern?", [d.value for d in Diet] + ["none"]),
    ("halal", "Halal preference?", ["strict", "meat_only", "none"]),
]


# Cuisine menu for the North American market. Deliberately no popularity priors: any numbers
# here would be invented, so fit `SoftSurvey.priors` from real order data when you have it.
NA_CUISINES = [
    "american", "bbq", "southern", "cajun", "mexican", "tex-mex", "canadian", "italian", "pizza",
    "chinese", "japanese", "korean", "thai", "vietnamese", "indian", "mediterranean", "greek",
    "middle_eastern", "caribbean", "french", "ethiopian",
]
DESSERT_TAGS = ["chocolate", "fruity", "creamy", "baked", "frozen", "nutty", "caramel"]
BEVERAGE_TAGS = ["coffee", "tea", "juice", "soda", "smoothie", "iced", "hot", "sparkling", "mocktail"]


def build_constraints(allergy_answers: dict[str, str], diet_answers: list[str], halal_answer: str) -> UserConstraints:
    """allergy_answers maps allergen value -> 'mild'|'moderate'|'severe' (asked per allergen)."""
    allergies = {Allergen(a): Severity[s.upper()] for a, s in allergy_answers.items()}
    diets = {Diet(d) for d in diet_answers if d != "none"}
    halal = {"strict": HalalLevel.STRICT, "meat_only": HalalLevel.MEAT_ONLY, "none": HalalLevel.NONE}[halal_answer]
    return UserConstraints(allergies=allergies, diets=diets, halal=halal)


def bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


@dataclass
class SoftSurvey:
    cuisines: list[str]
    priors: dict[str, float] = field(default_factory=dict)  # population popularity in (0,1)
    max_questions: int = 6
    prior_strength: float = 2.0
    answer_strength: float = 3.0
    min_gain: float = 0.3  # bits; stop when nothing is this uncertain

    def __post_init__(self) -> None:
        self._beta: dict[str, list[float]] = {}
        for c in self.cuisines:
            p = min(max(self.priors.get(c, 0.5), 0.05), 0.95)
            self._beta[c] = [p * self.prior_strength, (1 - p) * self.prior_strength]
        self._asked: list[str] = []
        self._declared: dict[str, bool] = {}  # cuisine -> liked (True) / avoided (False)

    def declare(self, liked: tuple = (), avoided: tuple = ()) -> None:
        """Apply cuisines the user named directly. Unknown names raise, so typos are not silently dropped."""
        liked = [c.lower() for c in liked]
        avoided = [c.lower() for c in avoided]
        unknown = sorted({c for c in liked + avoided if c not in self._beta})
        if unknown:
            raise ValueError(f"unknown cuisines: {unknown}")
        both = sorted(set(liked) & set(avoided))
        if both:
            raise ValueError(f"cuisines both liked and avoided: {both}")
        for c in liked:
            self._declared[c] = True
            self._beta[c][0] += 2 * self.answer_strength
        for c in avoided:
            self._declared[c] = False
            self._beta[c][1] += 2 * self.answer_strength

    def belief(self, cuisine: str) -> float:
        a, b = self._beta[cuisine]
        return a / (a + b)

    def expected_gain(self, cuisine: str) -> float:
        return bernoulli_entropy(self.belief(cuisine))

    def next_question(self) -> Optional[str]:
        if len(self._asked) >= self.max_questions:
            return None
        remaining = [c for c in self.cuisines if c not in self._asked and c not in self._declared]
        if not remaining:
            return None
        best = max(remaining, key=lambda c: (self.expected_gain(c), c))
        return best if self.expected_gain(best) >= self.min_gain else None

    def answer(self, cuisine: str, likes: bool) -> None:
        self._asked.append(cuisine)
        self._beta[cuisine][0 if likes else 1] += self.answer_strength

    def to_preferences(self, spice_tolerance: int = 2, adventurousness: float = 0.5) -> SoftPreferences:
        return SoftPreferences(
            {c: round(self.belief(c), 3) for c in self.cuisines}, spice_tolerance, adventurousness,
            cuisines_avoided=frozenset(c for c, liked in self._declared.items() if not liked),
        )


def _affinity(likes: tuple, dislikes: tuple, universe: list[str], kind: str) -> dict[str, float]:
    likes, dislikes = [t.lower() for t in likes], [t.lower() for t in dislikes]
    bad = sorted({t for t in likes + dislikes if t not in universe})
    if bad:
        raise ValueError(f"unknown {kind} tags: {bad}")
    clash = sorted(set(likes) & set(dislikes))
    if clash:
        raise ValueError(f"{kind} tags both liked and disliked: {clash}")
    return {**{t: 0.9 for t in likes}, **{t: 0.1 for t in dislikes}}


def build_taste_preferences(base: SoftPreferences, *, sweet_tooth: float = 0.5,
                            dessert_likes: tuple = (), dessert_dislikes: tuple = (),
                            beverage_likes: tuple = (), beverage_dislikes: tuple = (),
                            caffeine_ok: bool = True) -> SoftPreferences:
    """Return `base` plus sweet / dessert / beverage tastes. Built only from the user's answers."""
    if not 0.0 <= sweet_tooth <= 1.0:
        raise ValueError("sweet_tooth must be between 0 and 1")
    return SoftPreferences(
        cuisines=dict(base.cuisines), spice_tolerance=base.spice_tolerance,
        adventurousness=base.adventurousness, cuisines_avoided=base.cuisines_avoided,
        sweet_tooth=sweet_tooth,
        dessert_tags=_affinity(dessert_likes, dessert_dislikes, DESSERT_TAGS, "dessert"),
        beverage_tags=_affinity(beverage_likes, beverage_dislikes, BEVERAGE_TAGS, "beverage"),
        caffeine_ok=caffeine_ok,
    )
