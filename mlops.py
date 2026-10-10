"""Feedback, incident handling and the deployment gate.

Hard gate: a candidate pipeline is deployable only if it produces ZERO safety
violations on the safety suite. Business metrics are checked afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .knowledge_graph import KnowledgeGraph
from .safety import is_displayable, label_dish
from .schema import Allergen, Dish, IngredientMention, Restaurant, Severity, UserConstraints


@dataclass
class FeedbackEvent:
    user_id: str
    dish_id: str
    kind: str  # view | click | order | abandon | rating | allergy_incident
    value: Optional[float] = None  # rating, etc.
    allergen: Optional[Allergen] = None  # for allergy_incident


class IncidentRegistry:
    """Allergy-incident reports quarantine a dish for that allergen immediately,
    without waiting for the weekly retrain. Pass `.as_mapping()` to the safety layer."""

    def __init__(self) -> None:
        self._by_dish: dict[str, set] = {}

    def record(self, event: FeedbackEvent) -> None:
        if event.kind != "allergy_incident" or event.allergen is None:
            raise ValueError("not an allergy incident")
        self._by_dish.setdefault(event.dish_id, set()).add(event.allergen)

    def as_mapping(self) -> dict[str, frozenset]:
        return {k: frozenset(v) for k, v in self._by_dish.items()}


@dataclass
class SafetyCase:
    name: str
    constraints: UserConstraints
    dish: Dish
    restaurant: Restaurant
    must_be_hidden: bool = True


def kg_coverage_cases(kg: KnowledgeGraph, restaurant: Restaurant) -> list[SafetyCase]:
    """For every allergen-bearing ingredient, a user allergic to it must never see a dish made of it."""
    cases = []
    for node in kg.nodes:
        for allergen in node.allergens:
            dish = Dish(f"kg:{node.name}", restaurant.id, node.name, 1.0,
                        [IngredientMention(node.name)], ingredients_complete=True)
            cases.append(SafetyCase(f"{node.name}->{allergen.value}",
                                    UserConstraints(allergies={allergen: Severity.SEVERE}), dish, restaurant))
    return cases


def run_safety_suite(displayable: Callable[[Dish, Restaurant, UserConstraints], bool],
                     cases: Sequence[SafetyCase]) -> list[str]:
    """`displayable` is the full candidate pipeline's final visibility decision. Returns violation names."""
    return [c.name for c in cases if c.must_be_hidden and displayable(c.dish, c.restaurant, c.constraints)]


def default_displayable(kg: KnowledgeGraph):
    return lambda dish, rest, constraints: is_displayable(label_dish(dish, rest, constraints, kg), constraints)


def conversion_rate(events: Sequence[FeedbackEvent]) -> float:
    views = sum(e.kind == "view" for e in events)
    orders = sum(e.kind == "order" for e in events)
    return orders / views if views else 0.0


@dataclass
class GateReport:
    passed: bool
    reasons: list = field(default_factory=list)


def deployment_gate(violations: Sequence[str], baseline_conversion: float, candidate_conversion: float,
                    baseline_convergence_s: float, candidate_convergence_s: float,
                    max_conversion_drop: float = 0.02, max_convergence_slowdown: float = 1.10) -> GateReport:
    reasons = []
    if violations:  # hard gate, never overridden by business metrics
        reasons.append(f"safety violations: {len(violations)} (first: {violations[0]})")
    if candidate_conversion < baseline_conversion - max_conversion_drop:
        reasons.append("conversion regressed beyond tolerance")
    if candidate_convergence_s > baseline_convergence_s * max_convergence_slowdown:
        reasons.append("group convergence time regressed beyond tolerance")
    return GateReport(not reasons, reasons)
