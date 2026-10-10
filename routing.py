"""Order routing and redirection.

`MenuProvider` is the seam between this system and delivery sources. Implement
it per source (official partner API, ingestion pipeline, manual data) and the
rest of the system never changes. Deep-link URL templates are configuration:
real DoorDash/Uber Eats link formats and API access depend on your partner
agreement, so none are hard-coded here.

Before any link is produced, every selected dish is re-checked against its
eater's hard constraints (menus change between recommendation and checkout).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence
from urllib.parse import quote

from .group_safety import GroupPlan, Member
from .knowledge_graph import KnowledgeGraph
from .safety import is_displayable, label_dish
from .schema import Dish, Restaurant, UserConstraints

VALUE_OF_TIME_PER_MIN = 0.10  # currency units; tune per market


@dataclass
class Quote:
    provider: str
    restaurant_id: str
    available: bool
    item_total: float
    delivery_fee: float
    service_fee: float
    eta_min: float
    deep_link: Optional[str] = None

    @property
    def total_cost(self) -> float:
        return self.item_total + self.delivery_fee + self.service_fee


class MenuProvider(ABC):
    name: str

    @abstractmethod
    def quote(self, restaurant_id: str, dish_ids: Sequence[str], location: tuple[float, float]) -> Optional[Quote]:
        """Availability, price, fees and ETA for these items, or None if unsupported."""


class StaticProvider(MenuProvider):
    """In-memory provider for tests/demos: {restaurant_id: (available, fees, eta, {dish_id: price})}."""

    def __init__(self, name: str, data: Mapping[str, dict]) -> None:
        self.name, self._data = name, data

    def quote(self, restaurant_id, dish_ids, location):
        row = self._data.get(restaurant_id)
        if row is None:
            return None
        prices = row["prices"]
        available = row.get("available", True) and all(d in prices for d in dish_ids)
        return Quote(self.name, restaurant_id, available, sum(prices.get(d, 0.0) for d in dish_ids),
                     row.get("delivery_fee", 0.0), row.get("service_fee", 0.0), row.get("eta_min", 30.0))


def build_deep_link(templates: Mapping[str, str], quote_: Quote, dish_ids: Sequence[str]) -> Optional[str]:
    """templates: {"provider": "https://.../store/{restaurant_id}?items={items}"} from partner docs."""
    template = templates.get(quote_.provider)
    if template is None:
        return None
    return template.format(restaurant_id=quote(quote_.restaurant_id), items=quote(",".join(dish_ids)))


def rank_quotes(providers: Sequence[MenuProvider], restaurant_id: str, dish_ids: Sequence[str],
                location: tuple[float, float], templates: Optional[Mapping[str, str]] = None,
                value_of_time: float = VALUE_OF_TIME_PER_MIN) -> list[Quote]:
    quotes = []
    for p in providers:
        q = p.quote(restaurant_id, dish_ids, location)
        if q is not None and q.available:
            q.deep_link = build_deep_link(templates or {}, q, dish_ids)
            quotes.append(q)
    return sorted(quotes, key=lambda q: (q.total_cost + value_of_time * q.eta_min, q.provider))


class UnsafeOrderError(Exception):
    pass


def verify_order_safety(selections: Mapping[str, Sequence[Dish]], members: Mapping[str, UserConstraints],
                        restaurants: Mapping[str, Restaurant], kg: KnowledgeGraph,
                        incident_allergens: Optional[Mapping[str, frozenset]] = None) -> None:
    """selections: member id -> dishes they will eat. Raises if anything fails that member's constraints."""
    for mid, dishes in selections.items():
        for d in dishes:
            res = label_dish(d, restaurants.get(d.restaurant_id), members[mid], kg, incident_allergens=incident_allergens)
            if not is_displayable(res, members[mid]):
                raise UnsafeOrderError(f"{d.id} is not safe for {mid}: {res.reasons}")


@dataclass
class OrderIntent:
    restaurant_id: str
    eaters: dict  # member id -> [dish ids]
    quotes: list = field(default_factory=list)  # ranked Quotes
    warnings: dict = field(default_factory=dict)  # member id -> [warning codes]


def build_group_orders(
    picks: Mapping[str, Sequence[Dish]],  # member id -> dishes chosen
    members: Sequence[Member],
    plan: GroupPlan,
    restaurants: Mapping[str, Restaurant],
    kg: KnowledgeGraph,
    providers: Sequence[MenuProvider],
    location: tuple[float, float],
    templates: Optional[Mapping[str, str]] = None,
    incident_allergens: Optional[Mapping[str, frozenset]] = None,
    split: Optional[bool] = None,
) -> list[OrderIntent]:
    """One cart per restaurant (single-vendor group cart), or one per member when
    `split` is True. Re-validates safety first and raises UnsafeOrderError on any failure."""
    verify_order_safety(picks, {m.id: m.constraints for m in members}, restaurants, kg, incident_allergens)
    by_rest: dict[str, dict[str, list[str]]] = {}
    for mid, dishes in picks.items():
        for d in dishes:
            by_rest.setdefault(d.restaurant_id, {}).setdefault(mid, []).append(d.id)
    intents = []
    for rid, eaters in sorted(by_rest.items()):
        groups = [{m: ids} for m, ids in eaters.items()] if split else [eaters]
        for group in groups:
            all_ids = [i for ids in group.values() for i in ids]
            intents.append(OrderIntent(rid, group, rank_quotes(providers, rid, all_ids, location, templates),
                                       {m: plan.warnings.get(m, []) for m in group}))
    return intents
