"""Deterministic safety layer. No ML score ever reaches this module.

label_dish() returns SAFE / UNKNOWN / UNSAFE for one (dish, restaurant, user).
Rules:
  * Unresolved ingredient names, incomplete ingredient lists and source-dependent
    ingredients (lecithin, gelatin...) yield UNKNOWN, never SAFE.
  * A definite allergen is UNSAFE. For SEVERE allergies even a low-confidence
    NLP detection is UNSAFE; for lesser severities a detection below
    DETECT_CONFIRM becomes UNKNOWN.
  * is_displayable() hides UNKNOWN for users with a severe allergy or strict Halal.
  * needs_verification drives the "Verify with restaurant" label. For users with no hard
    constraints it fires only for unidentified ingredients or missing ingredient data;
    `verification_warning()` turns the result into the text shown to the user.
  * Beverages are shown only when the provider confirms `alcohol_free` and no ingredient
    resolves to alcohol. Missing data fails closed (UNSAFE for everyone).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from .knowledge_graph import KnowledgeGraph, Resolution
from .schema import (
    Dish, HalalLevel, Label, MeatSource, Restaurant, Severity, Trait as T, UserConstraints, Diet,
)

VERIFY_THRESHOLD = 0.9  # extraction confidence below this => "Verify with restaurant"
DETECT_CONFIRM = 0.5  # detections below this are UNKNOWN (non-severe) rather than UNSAFE

_DIET_BANNED = {
    Diet.VEGAN: {T.MEAT, T.PORK, T.PORK_POSSIBLE, T.FISH_FLESH, T.SHELLFISH_FLESH, T.ANIMAL_PRODUCT},
    Diet.VEGETARIAN: {T.MEAT, T.PORK, T.PORK_POSSIBLE, T.FISH_FLESH, T.SHELLFISH_FLESH},
}


@dataclass
class SafetyResult:
    label: Label
    reasons: list[str] = field(default_factory=list)
    needs_verification: bool = False
    warnings: list[str] = field(default_factory=list)
    # Ingredient names the knowledge graph could not identify (normalized).
    unknown_ingredients: list[str] = field(default_factory=list)
    no_ingredient_data: bool = False  # dish has no ingredient list at all


def is_displayable(result: SafetyResult, constraints: UserConstraints) -> bool:
    if result.label == Label.SAFE:
        return True
    if result.label == Label.UNKNOWN:
        return not constraints.exclude_unknown
    return False


Evidence = list  # list[tuple[IngredientMention, Resolution]]


def _max_conf(evidence: Evidence, pred: Callable[[Resolution], bool]) -> Optional[float]:
    confs = [m.confidence for m, r in evidence if pred(r)]
    return max(confs) if confs else None


def _effective_meat_source(restaurant: Optional[Restaurant]) -> MeatSource:
    if restaurant is None:
        return MeatSource.UNKNOWN
    if restaurant.halal_certified is True:
        # certified kitchen implies halal meat; conflicting data is not trusted
        return MeatSource.UNKNOWN if restaurant.meat_source == MeatSource.NON_HALAL else MeatSource.HALAL
    return restaurant.meat_source


def _halal(restaurant, level, evidence, unresolved, incomplete) -> list[tuple[Label, str]]:
    out: list[tuple[Label, str]] = []
    if level == HalalLevel.NONE:
        return out
    source = _effective_meat_source(restaurant)
    certified = restaurant.halal_certified if restaurant else None

    if restaurant is not None and restaurant.halal_certified is True and restaurant.meat_source == MeatSource.NON_HALAL:
        out.append((Label.UNKNOWN, "halal:conflicting_restaurant_data"))

    if _max_conf(evidence, lambda r: T.PORK in r.traits) is not None:
        out.append((Label.UNSAFE, "halal:pork"))  # any confidence: conservative

    if _max_conf(evidence, lambda r: T.MEAT in r.traits) is not None:
        if source == MeatSource.NON_HALAL:
            out.append((Label.UNSAFE, "halal:non_halal_meat"))
        elif source == MeatSource.UNKNOWN:
            out.append((Label.UNKNOWN, "halal:meat_source_unknown"))
    elif source != MeatSource.HALAL and (incomplete or unresolved):
        out.append((Label.UNKNOWN, "halal:meat_presence_unknown"))

    if level == HalalLevel.STRICT:
        if _max_conf(evidence, lambda r: T.ALCOHOL in r.traits) is not None:
            out.append((Label.UNSAFE, "halal:alcohol"))
        if _max_conf(evidence, lambda r: T.PORK_POSSIBLE in r.traits) is not None:
            if certified is True:
                pass
            elif certified is False:
                out.append((Label.UNSAFE, "halal:possible_pork_derivative"))
            else:
                out.append((Label.UNKNOWN, "halal:possible_pork_derivative_unverified"))
        if certified is not True:
            if restaurant is None or restaurant.alcohol_in_sauces is None:
                out.append((Label.UNKNOWN, "halal:alcohol_in_sauces_unknown"))
            elif restaurant.alcohol_in_sauces:
                out.append((Label.UNSAFE, "halal:alcohol_in_sauces"))
            if incomplete or unresolved:
                out.append((Label.UNKNOWN, "halal:hidden_ingredients_unverified"))
    return out


def label_dish(
    dish: Dish,
    restaurant: Optional[Restaurant],
    constraints: UserConstraints,
    kg: KnowledgeGraph,
    *,
    verify_threshold: float = VERIFY_THRESHOLD,
    incident_allergens: Optional[Mapping[str, frozenset]] = None,
) -> SafetyResult:
    evidence: Evidence = [(m, kg.expand(m.text)) for m in dish.ingredients]
    unresolved = sorted({u for _, r in evidence for u in r.unresolved})
    incomplete = (not dish.ingredients_complete) or not dish.ingredients
    result = SafetyResult(Label.SAFE, unknown_ingredients=unresolved, no_ingredient_data=not dish.ingredients)

    def bump(label: Label, reason: str) -> None:
        result.label = max(result.label, label)
        if reason not in result.reasons:
            result.reasons.append(reason)

    def data_gaps() -> None:
        if unresolved:
            bump(Label.UNKNOWN, "unresolved_ingredient:" + ",".join(unresolved))
        if incomplete:
            bump(Label.UNKNOWN, "incomplete_ingredient_data")

    # --- beverage policy: only provably alcohol-free drinks are ever recommended ---
    if dish.category == "beverage":
        if dish.alcohol_free is not True:
            bump(Label.UNSAFE, "beverage:not_confirmed_alcohol_free")
        if _max_conf(evidence, lambda r: T.ALCOHOL in r.traits) is not None:
            bump(Label.UNSAFE, "beverage:contains_alcohol")

    # --- allergies ---
    for allergen, severity in constraints.allergies.items():
        severe = severity == Severity.SEVERE
        if incident_allergens and allergen in incident_allergens.get(dish.id, frozenset()):
            bump(Label.UNSAFE, f"incident_report:{allergen.value}")
        conf = _max_conf(evidence, lambda r, a=allergen: a in r.allergens)
        if conf is not None:
            bump(Label.UNSAFE if (severe or conf >= DETECT_CONFIRM) else Label.UNKNOWN, f"contains:{allergen.value}")
        elif _max_conf(evidence, lambda r, a=allergen: a in r.possible) is not None:
            bump(Label.UNKNOWN, f"may_contain:{allergen.value}")
        if severe and restaurant is not None and allergen in restaurant.shared_kitchen_allergens:
            warning = f"cross_contact:{allergen.value}"
            if warning not in result.warnings:
                result.warnings.append(warning)
    if constraints.allergies:
        data_gaps()

    # --- diets ---
    for diet in constraints.diets:
        for trait in _DIET_BANNED[diet]:
            conf = _max_conf(evidence, lambda r, t=trait: t in r.traits)
            if conf is not None:
                bump(Label.UNSAFE if conf >= DETECT_CONFIRM else Label.UNKNOWN, f"diet_{diet.value}:{trait.value}")
    if constraints.diets:
        data_gaps()

    # --- halal ---
    for label, reason in _halal(restaurant, constraints.halal, evidence, unresolved, incomplete):
        bump(label, reason)

    unknown_data = bool(unresolved) or not dish.ingredients
    if constraints.is_empty:
        # Nothing hard is at stake, so only unidentified or missing ingredients warrant a prompt.
        result.needs_verification = unknown_data
    else:
        result.needs_verification = (
            incomplete or unknown_data or any(m.confidence < verify_threshold for m, _ in evidence)
        )
    return result


_MAX_LISTED = 5


def verification_warning(result: SafetyResult, constraints: Optional[UserConstraints] = None) -> Optional[str]:
    """User-facing text for a result, or None when there is nothing to verify."""
    if not result.needs_verification:
        return None
    parts = []
    if result.unknown_ingredients:
        names = result.unknown_ingredients[:_MAX_LISTED]
        more = len(result.unknown_ingredients) - len(names)
        listed = ", ".join(names) + (f" and {more} more" if more > 0 else "")
        parts.append(f"We could not identify these ingredients: {listed}.")
    if result.no_ingredient_data:
        parts.append("No ingredient information is available for this item.")
    if not parts:
        parts.append("The ingredient list may be incomplete or inferred.")
    text = "Verify with the restaurant before ordering. " + " ".join(parts)
    if constraints is not None and not constraints.is_empty:
        text += " This is a software filter, not a medical guarantee."
    return text


def safe_catalog(dishes, restaurants: Mapping[str, Restaurant], constraints: UserConstraints,
                 kg: KnowledgeGraph, **kwargs):
    """Yield (dish, SafetyResult) for every dish the user may be shown."""
    for dish in dishes:
        res = label_dish(dish, restaurants.get(dish.restaurant_id), constraints, kg, **kwargs)
        if is_displayable(res, constraints):
            yield dish, res
