"""Candidate retrieval.

1. Operational filters: delivery radius, opening hours, budget, ETA.
2. Two-tower style similarity: user tower (soft prefs + context) vs dish tower.
   Baseline towers use deterministic feature hashing so the system runs without
   trained weights; swap in learned towers (PyTorch) behind `embed_user` /
   `embed_dish`, and `BruteForceIndex` for pgvector/ANN behind `VectorIndex`.

Retrieval receives ONLY dishes that already passed the safety filter.
"""
from __future__ import annotations

import zlib
from typing import Mapping, Protocol, Sequence

import numpy as np

from .group_safety import haversine_km
from .schema import Context, Dish, Restaurant, SoftPreferences

DIM = 64
COURIER_KMH = 20.0


def _h(token: str) -> int:
    return zlib.crc32(token.encode()) % DIM


def embed_user(soft: SoftPreferences) -> np.ndarray:
    v = np.zeros(DIM)
    for cuisine, w in soft.cuisines.items():
        v[_h(f"cuisine:{cuisine}")] += w - 0.5  # dislikes push away
    v[_h(f"spice:{soft.spice_tolerance}")] += 0.5
    for prefs in (soft.dessert_tags, soft.beverage_tags):
        for tag, w in prefs.items():
            v[_h(f"tag:{tag}")] += w - 0.5
    return v


def embed_dish(dish: Dish) -> np.ndarray:
    if dish.embedding is not None:
        return np.asarray(dish.embedding, dtype=float)
    v = np.zeros(DIM)
    v[_h(f"cuisine:{dish.cuisine}")] += 1.0
    v[_h(f"spice:{dish.spice_level}")] += 0.5
    for tag in dish.tags:
        v[_h(f"tag:{tag}")] += 0.5
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


def is_open(r: Restaurant, hour: int) -> bool:
    if r.open_hour <= r.close_hour:
        return r.open_hour <= hour < r.close_hour
    return hour >= r.open_hour or hour < r.close_hour  # overnight


def eta_minutes(r: Restaurant, ctx: Context) -> float:
    km = haversine_km(ctx.lat, ctx.lon, r.lat, r.lon)
    return r.prep_minutes + km / COURIER_KMH * 60.0


def operational_filter(dishes: Sequence[Dish], restaurants: Mapping[str, Restaurant], ctx: Context) -> list[Dish]:
    out = []
    for d in dishes:
        r = restaurants.get(d.restaurant_id)
        if r is None or d.price > ctx.budget or not is_open(r, ctx.hour):
            continue
        if haversine_km(ctx.lat, ctx.lon, r.lat, r.lon) > r.delivery_radius_km:
            continue
        if eta_minutes(r, ctx) > ctx.max_eta_min:
            continue
        out.append(d)
    return out


class VectorIndex(Protocol):
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...


class BruteForceIndex:
    """Exact cosine search. Replace with pgvector / HNSW at scale."""

    def __init__(self, dishes: Sequence[Dish]) -> None:
        self.ids = [d.id for d in dishes]
        self.mat = np.array([embed_dish(d) for d in dishes]) if dishes else np.zeros((0, DIM))

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        norms = np.linalg.norm(self.mat, axis=1) * (np.linalg.norm(query) or 1.0)
        sims = np.divide(self.mat @ query, norms, out=np.zeros(len(self.ids)), where=norms > 0)
        order = np.argsort(-sims, kind="stable")[:k]
        return [(self.ids[i], float(sims[i])) for i in order]


def retrieve(safe_dishes: Sequence[Dish], restaurants: Mapping[str, Restaurant], soft: SoftPreferences,
             ctx: Context, k: int = 200) -> list[tuple[Dish, float]]:
    candidates = operational_filter(safe_dishes, restaurants, ctx)
    index = BruteForceIndex(candidates)
    by_id = {d.id: d for d in candidates}
    return [(by_id[i], s) for i, s in index.search(embed_user(soft), k)]
