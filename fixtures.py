from foodrec.knowledge_graph import build_default_graph
from foodrec.schema import (
    Allergen, Diet, Dish, HalalLevel, IngredientMention, MeatSource, Restaurant, Severity, UserConstraints,
)

KG = build_default_graph()


def dish(did="d1", ings=(), complete=True, rid="r1", **kw):
    """Build a dish. `ings` items are names or (name, confidence) tuples."""
    mentions = []
    for i in ings:
        name, conf = (i, 1.0) if isinstance(i, str) else i
        mentions.append(IngredientMention(name, conf, "restaurant" if conf == 1.0 else "nlp"))
    kw.setdefault("price", 10.0)
    return Dish(id=did, restaurant_id=rid, name=kw.pop("name", did), ingredients=mentions,
                ingredients_complete=complete, **kw)


def rest(rid="r1", **kw):
    kw.setdefault("lat", 40.0)
    kw.setdefault("lon", -74.0)
    return Restaurant(id=rid, name=rid, **kw)


def allergic(*pairs):
    """allergic((Allergen.DAIRY, Severity.SEVERE), ...)"""
    return UserConstraints(allergies=dict(pairs))


SEVERE_DAIRY = allergic((Allergen.DAIRY, Severity.SEVERE))
HALAL_CERTIFIED = dict(halal_certified=True, meat_source=MeatSource.HALAL, alcohol_in_sauces=False)
