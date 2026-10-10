"""Core data schema.

Design rule: hard constraints (allergies, diet, Halal) live in `UserConstraints`
and are consumed ONLY by the safety layer. Nothing in `SoftPreferences` or the
ranking features may be derived from them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional


class Allergen(str, Enum):
    DAIRY = "dairy"
    EGG = "egg"
    FISH = "fish"
    SHELLFISH = "shellfish"  # crustaceans and molluscs
    TREE_NUT = "tree_nut"
    PEANUT = "peanut"
    WHEAT_GLUTEN = "wheat_gluten"  # wheat plus barley/rye/malt, mapped conservatively
    SOY = "soy"
    SESAME = "sesame"


class Severity(IntEnum):
    MILD = 1
    MODERATE = 2
    SEVERE = 3


class HalalLevel(IntEnum):
    NONE = 0
    MEAT_ONLY = 1  # only meat must be halal-sourced
    STRICT = 2  # meat, no pork derivatives, no alcohol


class Diet(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"


class Label(IntEnum):
    """Ordered by badness so `max()` gives the worst label."""

    SAFE = 0
    UNKNOWN = 1
    UNSAFE = 2


class MeatSource(str, Enum):
    HALAL = "halal"
    NON_HALAL = "non_halal"
    UNKNOWN = "unknown"


class Trait(str, Enum):
    MEAT = "meat"  # land-animal flesh or derivatives (incl. gelatin)
    PORK = "pork"
    PORK_POSSIBLE = "pork_possible"  # ambiguous animal source (gelatin, sausage)
    FISH_FLESH = "fish_flesh"
    SHELLFISH_FLESH = "shellfish_flesh"
    ALCOHOL = "alcohol"
    ANIMAL_PRODUCT = "animal_product"  # dairy, egg, honey
    CAFFEINE = "caffeine"  # coffee, true tea, matcha, cola (taste filter only, never a safety label)


@dataclass(frozen=True)
class IngredientMention:
    text: str
    confidence: float = 1.0
    source: str = "restaurant"  # "restaurant" (structured list) or "nlp" (inferred)


@dataclass
class Dish:
    id: str
    restaurant_id: str
    name: str
    price: float
    ingredients: list[IngredientMention] = field(default_factory=list)
    # True only when the full ingredient list came from the restaurant/provider.
    ingredients_complete: bool = False
    description: str = ""
    cuisine: str = ""
    spice_level: int = 0  # 0-5
    calories: int = 0
    portion: float = 1.0  # relative size, ~0.5 small .. 1.5 large
    is_hot: bool = True  # served hot (soup, curry) vs cold (salad)
    embedding: Optional[list[float]] = None
    category: str = "main"  # "main" | "dessert" | "beverage"
    sweetness: int = 0  # 0-5, used for desserts and beverages
    tags: tuple[str, ...] = ()  # taste tags, e.g. ("chocolate", "baked") or ("coffee", "iced")
    # Beverages only. Must be True (provider-confirmed) for the dish to ever be shown;
    # None/False fails closed. See safety.label_dish.
    alcohol_free: Optional[bool] = None
    caffeinated: Optional[bool] = None  # None = unknown; the knowledge graph may also infer it


@dataclass
class Restaurant:
    id: str
    name: str
    lat: float
    lon: float
    cuisines: tuple[str, ...] = ()
    halal_certified: Optional[bool] = None  # whole-kitchen certification; None = unknown
    meat_source: MeatSource = MeatSource.UNKNOWN
    alcohol_in_sauces: Optional[bool] = None  # None = unknown
    shared_kitchen_allergens: frozenset[Allergen] = frozenset()  # declared cross-contact
    delivery_radius_km: float = 5.0
    open_hour: int = 0  # local hour, inclusive
    close_hour: int = 24  # local hour, exclusive; may be < open_hour for overnight
    prep_minutes: int = 15


@dataclass
class UserConstraints:
    """HARD constraints. Explicitly collected, never learned or inferred."""

    allergies: dict[Allergen, Severity] = field(default_factory=dict)
    diets: set[Diet] = field(default_factory=set)
    halal: HalalLevel = HalalLevel.NONE

    @property
    def has_severe_allergy(self) -> bool:
        return any(s == Severity.SEVERE for s in self.allergies.values())

    @property
    def is_empty(self) -> bool:
        return not self.allergies and not self.diets and self.halal == HalalLevel.NONE

    @property
    def exclude_unknown(self) -> bool:
        """Unknown dishes are hidden for severe allergies (spec) and strict Halal
        (assumption: strict Halal is a hard religious constraint)."""
        return self.has_severe_allergy or self.halal == HalalLevel.STRICT


@dataclass
class SoftPreferences:
    """Learned tastes. Seeds the user embedding; contains no safety data."""

    cuisines: dict[str, float] = field(default_factory=dict)  # 0..1 affinity
    spice_tolerance: int = 2  # 0-5
    adventurousness: float = 0.5  # 0..1
    # Cuisines the user never wants suggested. A taste choice (applied after the safety
    # gate in taste.py), not a safety constraint, so it is not part of UserConstraints.
    cuisines_avoided: frozenset = frozenset()
    sweet_tooth: float = 0.5  # 0..1, general liking for sweet things
    dessert_tags: dict[str, float] = field(default_factory=dict)  # tag -> 0..1 affinity
    beverage_tags: dict[str, float] = field(default_factory=dict)
    caffeine_ok: bool = True  # False hides beverages known to contain caffeine


@dataclass
class UserProfile:
    id: str
    constraints: UserConstraints = field(default_factory=UserConstraints)
    soft: SoftPreferences = field(default_factory=SoftPreferences)
    consent_given: bool = False


@dataclass
class Context:
    lat: float
    lon: float
    budget: float  # max spend per dish
    hunger: float = 0.5  # 0 (snack) .. 1 (starving)
    temperature_c: float = 20.0
    raining: bool = False
    hour: int = 12
    max_eta_min: int = 60
    sweet_craving: float = 0.0  # 0 (none) .. 1 (strong craving right now)
