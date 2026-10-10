"""Group preference fusion over a SAFE shared pool.

The pool comes from group_safety.plan_group_order (mode SHARED), so every dish
is already safe for every member. This module only decides ordering, and the
session refuses any dish id that is not in the pool.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from .ranking import HeuristicRanker, Ranker, build_features
from .retrieval import cosine, embed_dish, embed_user
from .schema import Context, Dish, Restaurant, UserProfile

EPS = 1e-3


def minmax(raw: Mapping[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    lo, hi = min(raw.values()), max(raw.values())
    return {k: (EPS + (1 - EPS) * (v - lo) / (hi - lo)) if hi > lo else 1.0 for k, v in raw.items()}


def build_member_scores(members: Mapping[str, UserProfile], pool: Sequence[Dish],
                        restaurants: Mapping[str, Restaurant], ctx: Context,
                        ranker: Optional[Ranker] = None) -> dict[str, dict[str, float]]:
    """Per-member normalized scores in (0,1] for each pool dish."""
    ranker = ranker or HeuristicRanker()
    out = {}
    for mid, profile in members.items():
        uvec = embed_user(profile.soft)
        rows = [build_features(d, restaurants[d.restaurant_id], profile.soft, ctx, cosine(uvec, embed_dish(d)))
                for d in pool]
        out[mid] = minmax(dict(zip((d.id for d in pool), ranker.score(rows))))
    return out


class FairnessLedger:
    """Tracks cumulative regret per member so persistent losers get more weight."""

    def __init__(self, strength: float = 1.0) -> None:
        self.strength = strength
        self.regret: dict[str, float] = {}
        self.sessions: dict[str, int] = {}

    def weights(self, member_ids: Sequence[str]) -> dict[str, float]:
        avg = {m: self.regret.get(m, 0.0) / max(1, self.sessions.get(m, 0)) for m in member_ids}
        top = max(avg.values(), default=0.0)
        return {m: 1.0 + (self.strength * avg[m] / top if top > 0 else 0.0) for m in member_ids}

    def record(self, chosen: Mapping[str, float], best: Mapping[str, float]) -> None:
        for m in chosen:
            self.regret[m] = self.regret.get(m, 0.0) + max(0.0, best[m] - chosen[m])
            self.sessions[m] = self.sessions.get(m, 0) + 1


def aggregate(scores: Mapping[str, float], method: str = "nash", weights: Optional[Mapping[str, float]] = None) -> float:
    """Weighted least-misery or Nash welfare. Scores are in (0,1]; weight w>1 makes a
    member count MORE (s**w shrinks), which is how the fairness term helps past losers."""
    w = weights or {}
    powered = {m: max(s, EPS) ** w.get(m, 1.0) for m, s in scores.items()}
    if method == "least_misery":
        return min(powered.values())
    if method == "nash":
        total_w = sum(w.get(m, 1.0) for m in scores)
        return math.exp(sum(w.get(m, 1.0) * math.log(max(s, EPS)) for m, s in scores.items()) / total_w)
    raise ValueError(f"unknown method: {method}")


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


@dataclass
class GroupSession:
    member_ids: list[str]
    pool: list[Dish]
    base_scores: dict[str, dict[str, float]]  # member -> dish id -> (0,1]
    method: str = "nash"
    ledger: FairnessLedger = field(default_factory=FairnessLedger)
    rng: random.Random = field(default_factory=random.Random)
    near_optimal: float = 0.85
    page_size: int = 3
    vote_step: float = 0.6

    def __post_init__(self) -> None:
        self._dishes = {d.id: d for d in self.pool}
        self._adjust: dict[str, dict[str, float]] = {m: {} for m in self.member_ids}  # member -> cuisine -> logit shift
        self._votes: dict[str, dict[str, bool]] = {}
        self._shown: set[str] = set()

    # ---- scoring ----------------------------------------------------------
    def member_score(self, member: str, dish: Dish) -> float:
        base = self.base_scores[member][dish.id]
        shift = self._adjust[member].get(dish.cuisine, 0.0)
        return 1.0 / (1.0 + math.exp(-(_logit(base) + shift)))

    def group_score(self, dish: Dish) -> float:
        scores = {m: self.member_score(m, dish) for m in self.member_ids}
        return aggregate(scores, self.method, self.ledger.weights(self.member_ids))

    def per_person_match(self, dish: Dish) -> dict[str, int]:
        return {m: round(100 * self.member_score(m, dish)) for m in self.member_ids}

    # ---- shuffle ------------------------------------------------------------
    def shuffle(self) -> list[dict]:
        fresh = [d for d in self.pool if d.id not in self._shown]
        if len(fresh) < self.page_size:  # exhausted: start another pass
            self._shown.clear()
            fresh = list(self.pool)
        scored = {d.id: self.group_score(d) for d in fresh}
        best = max(scored.values(), default=0.0)
        near = [d for d in fresh if scored[d.id] >= self.near_optimal * best]
        picks: list[Dish] = []
        while near and len(picks) < self.page_size:
            distinct = [d for d in near if all((d.restaurant_id, d.cuisine) != (p.restaurant_id, p.cuisine) for p in picks)]
            choices = distinct or near
            pick = self.rng.choices(choices, weights=[scored[d.id] for d in choices], k=1)[0]
            picks.append(pick)
            near.remove(pick)
        if len(picks) < self.page_size:  # top up from the rest by score
            rest = sorted((d for d in fresh if d not in picks), key=lambda d: -scored[d.id])
            picks += rest[: self.page_size - len(picks)]
        self._shown.update(d.id for d in picks)
        return [{"dish_id": d.id, "name": d.name, "group_score": round(scored[d.id], 3),
                 "match": self.per_person_match(d)} for d in picks]

    # ---- voting -----------------------------------------------------------------
    def vote(self, member: str, dish_id: str, like: bool) -> None:
        if member not in self.member_ids:
            raise KeyError(f"unknown member: {member}")
        if dish_id not in self._dishes:
            raise KeyError(f"dish not in safe pool: {dish_id}")
        self._votes.setdefault(dish_id, {})[member] = like
        cuisine = self._dishes[dish_id].cuisine
        step = self.vote_step if like else -self.vote_step
        self._adjust[member][cuisine] = self._adjust[member].get(cuisine, 0.0) + step

    def consensus(self) -> Optional[str]:
        agreed = [d for d, v in self._votes.items()
                  if len(v) == len(self.member_ids) and all(v.values())]
        return max(agreed, key=lambda d: self.group_score(self._dishes[d]), default=None)

    def finalize(self, dish_id: str) -> dict[str, int]:
        if dish_id not in self._dishes:
            raise KeyError(f"dish not in safe pool: {dish_id}")
        dish = self._dishes[dish_id]
        chosen = {m: self.member_score(m, dish) for m in self.member_ids}
        best = {m: max(self.member_score(m, d) for d in self.pool) for m in self.member_ids}
        self.ledger.record(chosen, best)
        return self.per_person_match(dish)

    # ---- transport-agnostic event handler (wrapped by the WebSocket layer) ----------
    def handle_event(self, event: Mapping) -> dict:
        try:
            kind = event["type"]
            if kind == "reroll":
                return {"type": "options", "options": self.shuffle()}
            if kind == "vote":
                self.vote(event["member"], event["dish_id"], bool(event["like"]))
                return {"type": "vote_ack", "consensus": self.consensus()}
            if kind == "finalize":
                return {"type": "finalized", "dish_id": event["dish_id"], "match": self.finalize(event["dish_id"])}
            return {"type": "error", "message": f"unknown event type: {kind}"}
        except KeyError as exc:
            return {"type": "error", "message": str(exc)}
