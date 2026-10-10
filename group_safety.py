"""Group safety: shared pool, single-vendor and split-order planning.

Invariant (enforced by construction, tested exhaustively): every dish placed in
front of a member passes THAT member's own hard constraints.

Tiers, tried in order:
  SHARED         dishes safe for everyone (union of restrictions), any restaurant
  SINGLE_VENDOR  one restaurant where each member has >=1 personally safe dish
  SPLIT          up to `max_vendors` restaurants that together cover every member
  NONE           no safe plan; `unserved` lists who cannot be covered
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Mapping, Optional

from .knowledge_graph import KnowledgeGraph
from .safety import SafetyResult, is_displayable, label_dish
from .schema import Dish, Restaurant, Severity, UserConstraints


@dataclass
class Member:
    id: str
    constraints: UserConstraints


@dataclass
class GroupPlan:
    mode: str  # SHARED | SINGLE_VENDOR | SPLIT | NONE
    shared_pool: list = field(default_factory=list)  # list[Dish] (SHARED)
    # member id -> (restaurant id, ids of dishes safe for that member there)
    assignments: dict = field(default_factory=dict)
    warnings: dict = field(default_factory=dict)  # member id -> list[str]
    unserved: list = field(default_factory=list)
    note: str = ""


def merge_constraints(members: list[Member]) -> UserConstraints:
    """Union of all members' hard restrictions (strictest value wins)."""
    merged = UserConstraints()
    for m in members:
        for allergen, sev in m.constraints.allergies.items():
            merged.allergies[allergen] = max(merged.allergies.get(allergen, Severity.MILD), sev)
        merged.diets |= m.constraints.diets
        merged.halal = max(merged.halal, m.constraints.halal)
    return merged


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def _member_safe_ids(member, dishes_by_rest, restaurants, kg, incident) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for rid, dishes in dishes_by_rest.items():
        ok = []
        for d in dishes:
            res = label_dish(d, restaurants.get(rid), member.constraints, kg, incident_allergens=incident)
            if is_displayable(res, member.constraints):
                ok.append(d.id)
        out[rid] = ok
    return out


def _warnings_for(member: Member, restaurant: Optional[Restaurant], shared_kitchen: bool) -> list[str]:
    warns: list[str] = []
    severe = [a for a, s in member.constraints.allergies.items() if s == Severity.SEVERE]
    if severe and restaurant is not None:
        for a in severe:
            if a in restaurant.shared_kitchen_allergens:
                warns.append(f"cross_contact:{a.value}")
        if shared_kitchen:
            warns.append("confirm_cross_contact_practices_with_restaurant")
    return warns


def plan_group_order(
    members: list[Member],
    restaurants: Mapping[str, Restaurant],
    dishes: list[Dish],
    kg: KnowledgeGraph,
    *,
    group_location: Optional[tuple[float, float]] = None,
    max_vendors: int = 2,
    min_shared: int = 1,
    incident_allergens: Optional[Mapping[str, frozenset]] = None,
) -> GroupPlan:
    by_rest: dict[str, list[Dish]] = {}
    for d in dishes:
        if d.restaurant_id in restaurants:
            by_rest.setdefault(d.restaurant_id, []).append(d)

    safe = {m.id: _member_safe_ids(m, by_rest, restaurants, kg, incident_allergens) for m in members}

    # Tier 1: shared pool = dishes personally safe for EVERY member
    shared = [
        d for rid, ds in by_rest.items() for d in ds
        if all(d.id in safe[m.id][rid] for m in members)
    ]
    if len(shared) >= min_shared and members:
        warnings = {
            m.id: sorted({w for d in shared for w in _warnings_for(m, restaurants[d.restaurant_id], False)
                          if w.startswith("cross_contact")})
            for m in members
        }
        return GroupPlan("SHARED", shared_pool=shared, warnings=warnings)

    def dist(rid: str) -> float:
        if group_location is None:
            return 0.0
        r = restaurants[rid]
        return haversine_km(group_location[0], group_location[1], r.lat, r.lon)

    def serves(rid: str, m: Member) -> bool:
        return bool(safe[m.id][rid])

    # Tier 2: one restaurant where everyone has their own safe dish
    singles = sorted((rid for rid in by_rest if all(serves(rid, m) for m in members)),
                     key=lambda rid: (dist(rid), rid))
    if singles and members:
        rid = singles[0]
        assignments = {m.id: (rid, list(safe[m.id][rid])) for m in members}
        warnings = {m.id: _warnings_for(m, restaurants[rid], shared_kitchen=True) for m in members}
        return GroupPlan("SINGLE_VENDOR", assignments=assignments, warnings=warnings,
                         note="No common dish; each member orders their own safe dish.")

    # Tier 3: split across up to max_vendors restaurants
    unserved_anywhere = [m.id for m in members if not any(serves(rid, m) for rid in by_rest)]
    if unserved_anywhere:
        return GroupPlan("NONE", unserved=unserved_anywhere,
                         note="No restaurant has a dish that is safe for these members.")
    useful = sorted((rid for rid in by_rest if any(serves(rid, m) for m in members)),
                    key=lambda rid: (dist(rid), rid))
    for k in range(2, max_vendors + 1):
        best = None
        for combo in combinations(useful, k):
            if all(any(serves(rid, m) for rid in combo) for m in members):
                cost = sum(dist(rid) for rid in combo)
                if best is None or cost < best[0]:
                    best = (cost, combo)
        if best:
            combo = best[1]
            assignments, warnings = {}, {}
            for m in members:
                rid = min((r for r in combo if serves(r, m)), key=lambda r: (dist(r), r))
                assignments[m.id] = (rid, list(safe[m.id][rid]))
                warnings[m.id] = _warnings_for(m, restaurants[rid], shared_kitchen=True)
            return GroupPlan("SPLIT", assignments=assignments, warnings=warnings,
                             note=f"Split order across {len(combo)} restaurants.")
    return GroupPlan("NONE", note=f"Coverage would need more than {max_vendors} restaurants.")
