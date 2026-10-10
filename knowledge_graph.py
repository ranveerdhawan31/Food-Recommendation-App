"""Ingredient knowledge graph.

Maps ingredient names (and aliases) to definite allergens, *possible* allergens
(source-dependent, e.g. lecithin), dietary/Halal traits, and component
ingredients for composites (e.g. caesar dressing -> egg, anchovy, parmesan).

Unknown names are never assumed safe: `expand` reports them as unresolved.
This is a seed dataset; extend it from a curated source before production.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Allergen as A
from .schema import Trait as T


@dataclass(frozen=True)
class Node:
    name: str
    allergens: frozenset = frozenset()
    possible_allergens: frozenset = frozenset()
    traits: frozenset = frozenset()
    contains: tuple = ()


@dataclass
class Resolution:
    allergens: set = field(default_factory=set)
    possible: set = field(default_factory=set)
    traits: set = field(default_factory=set)
    unresolved: list = field(default_factory=list)

    def merge(self, other: "Resolution") -> None:
        self.allergens |= other.allergens
        self.possible |= other.possible
        self.traits |= other.traits
        self.unresolved += other.unresolved


def normalize(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"[^a-z0-9\s'-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


class KnowledgeGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._alias: dict[str, str] = {}

    # ---- building -------------------------------------------------------
    def add(self, name, aliases=(), allergens=(), possible=(), traits=(), contains=()):
        node = Node(name, frozenset(allergens), frozenset(possible), frozenset(traits), tuple(contains))
        self._nodes[name] = node
        self._alias[normalize(name)] = name
        for a in aliases:
            self._alias[normalize(a)] = name

    # ---- lookup ---------------------------------------------------------
    def lookup(self, text: str):
        key = normalize(text)
        if key in self._alias:
            return self._nodes[self._alias[key]]
        for suffix in ("es", "s"):  # naive plural fallback
            if key.endswith(suffix) and key[: -len(suffix)] in self._alias:
                return self._nodes[self._alias[key[: -len(suffix)]]]
        return None

    @property
    def nodes(self) -> list[Node]:
        return list(self._nodes.values())

    def all_names(self) -> list[str]:
        """Every known surface form, longest first (for text scanning)."""
        return sorted(self._alias, key=len, reverse=True)

    def expand(self, text: str, _seen: frozenset = frozenset()) -> Resolution:
        res = Resolution()
        node = self.lookup(text)
        if node is None:
            res.unresolved.append(normalize(text))
            return res
        if node.name in _seen:  # cycle guard
            return res
        res.allergens |= set(node.allergens)
        res.possible |= set(node.possible_allergens)
        res.traits |= set(node.traits)
        for child in node.contains:
            res.merge(self.expand(child, _seen | {node.name}))
        return res


def build_default_graph() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    add = kg.add

    # --- plain, allergen-free ingredients (known-safe) ---
    for n in ["rice", "tomato", "onion", "garlic", "ginger", "potato", "carrot", "lettuce",
              "cucumber", "chickpea", "lentil", "olive oil", "vegetable oil", "salt", "pepper",
              "cilantro", "lemon", "lime", "basil", "cumin", "turmeric", "chili", "bell pepper",
              "spinach", "mushroom", "bean", "corn", "coconut milk", "sugar", "water", "vinegar",
              "oats", "avocado", "cabbage", "broccoli", "eggplant", "peas", "mint", "paprika",
              "cinnamon", "coriander", "curry leaf", "tamarind", "lemongrass"]:
        add(n)

    # --- meat / pork ---
    for n in ["chicken", "beef", "lamb", "goat", "turkey", "duck", "veal", "mutton"]:
        add(n, traits=[T.MEAT])
    add("pork", traits=[T.MEAT, T.PORK])
    add("bacon", traits=[T.MEAT, T.PORK])
    add("ham", traits=[T.MEAT, T.PORK])
    add("lard", aliases=["pork fat"], traits=[T.MEAT, T.PORK])
    add("pepperoni", traits=[T.MEAT, T.PORK])
    add("sausage", aliases=["salami", "chorizo"], traits=[T.MEAT, T.PORK_POSSIBLE])
    add("gelatin", aliases=["gelatine"], traits=[T.MEAT, T.PORK_POSSIBLE])
    add("bone broth", aliases=["stock", "broth", "bouillon"], traits=[T.MEAT, T.PORK_POSSIBLE])
    add("chicken stock", aliases=["chicken broth"], traits=[T.MEAT])

    # --- fish / shellfish ---
    for n in ["fish", "salmon", "tuna", "cod", "anchovy", "sardine", "tilapia"]:
        add(n, allergens=[A.FISH], traits=[T.FISH_FLESH])
    add("fish sauce", allergens=[A.FISH], traits=[T.FISH_FLESH])
    add("worcestershire sauce", aliases=["worcestershire", "worcester sauce"],
        allergens=[A.FISH], possible=[A.SOY, A.WHEAT_GLUTEN], traits=[T.FISH_FLESH])
    for n in ["shrimp", "prawn", "crab", "lobster", "oyster", "mussel", "clam", "scallop", "squid"]:
        add(n, allergens=[A.SHELLFISH], traits=[T.SHELLFISH_FLESH])
    add("shrimp paste", allergens=[A.SHELLFISH], traits=[T.SHELLFISH_FLESH])
    add("oyster sauce", allergens=[A.SHELLFISH], possible=[A.SOY, A.WHEAT_GLUTEN], traits=[T.SHELLFISH_FLESH])

    # --- dairy ---
    dairy = [T.ANIMAL_PRODUCT]
    for n in ["milk", "butter", "ghee", "cream", "cheese", "parmesan", "paneer", "yogurt", "curd",
              "whey", "casein", "lactose", "buttermilk", "ice cream", "mozzarella", "feta"]:
        add(n, allergens=[A.DAIRY], traits=dairy)
    add("sodium caseinate", aliases=["caseinate", "calcium caseinate"], allergens=[A.DAIRY], traits=dairy)
    add("lactalbumin", allergens=[A.DAIRY], traits=dairy)

    # --- egg ---
    for n in ["egg", "albumin", "ovalbumin", "meringue", "egg yolk", "egg white"]:
        add(n, allergens=[A.EGG], traits=[T.ANIMAL_PRODUCT])
    add("mayonnaise", aliases=["mayo"], contains=["egg", "vegetable oil"])
    add("lecithin", aliases=["lecithin e322", "e322"], possible=[A.SOY, A.EGG])
    add("honey", traits=[T.ANIMAL_PRODUCT])

    # --- wheat / gluten ---
    for n in ["wheat", "flour", "bread", "pasta", "semolina", "couscous", "seitan", "breadcrumbs",
              "tortilla", "barley", "rye", "malt", "malt vinegar", "noodles", "dumpling wrapper"]:
        add(n, allergens=[A.WHEAT_GLUTEN])
    add("naan", contains=["flour", "yogurt"], possible=[A.EGG])

    # --- soy ---
    for n in ["soy", "soybean", "tofu", "tempeh", "edamame", "soy lecithin"]:
        add(n, allergens=[A.SOY])
    add("miso", allergens=[A.SOY], possible=[A.WHEAT_GLUTEN])
    add("soy sauce", aliases=["shoyu", "tamari"], allergens=[A.SOY, A.WHEAT_GLUTEN])

    # --- sesame ---
    add("sesame", aliases=["sesame seed"], allergens=[A.SESAME])
    add("sesame oil", allergens=[A.SESAME])
    add("tahini", allergens=[A.SESAME])
    add("hummus", contains=["chickpea", "tahini", "olive oil", "lemon", "garlic"])

    # --- nuts / peanut ---
    for n in ["almond", "cashew", "walnut", "pecan", "pistachio", "hazelnut", "pine nut", "macadamia"]:
        add(n, allergens=[A.TREE_NUT])
    add("marzipan", aliases=["almond paste"], allergens=[A.TREE_NUT], contains=["sugar"])
    add("pesto", contains=["basil", "pine nut", "parmesan", "olive oil", "garlic"])
    add("peanut", aliases=["groundnut"], allergens=[A.PEANUT])
    add("peanut butter", allergens=[A.PEANUT])
    add("peanut oil", aliases=["groundnut oil"], allergens=[A.PEANUT])  # conservative: treat refined oil as peanut
    add("satay sauce", contains=["peanut", "soy sauce", "coconut milk"])
    add("caesar dressing", contains=["egg", "anchovy", "parmesan", "olive oil", "garlic"])

    # --- alcohol ---
    for n in ["wine", "red wine", "white wine", "cooking wine", "beer", "rum", "sake", "brandy",
              "vodka", "bourbon", "marsala", "mirin"]:
        add(n, traits=[T.ALCOHOL])
    for n in ["liqueur", "amaretto", "kahlua", "champagne", "prosecco", "whiskey", "whisky", "gin",
              "tequila", "hard cider", "triple sec", "cointreau", "grand marnier", "sherry", "port"]:
        add(n, traits=[T.ALCOHOL])
    add("irish cream", aliases=["baileys"], allergens=[A.DAIRY], traits=[T.ALCOHOL, T.ANIMAL_PRODUCT])
    # Conservative: both contain a small but real amount of alcohol.
    add("vanilla extract", traits=[T.ALCOHOL])
    add("kombucha", traits=[T.ALCOHOL])

    # --- dessert ingredients ---
    for n in ["fruit", "apple", "banana", "strawberry", "blueberry", "raspberry", "blackberry", "peach",
              "cherry", "mango", "orange", "pineapple", "pumpkin", "cranberry", "grape", "pear", "coconut",
              "vanilla", "vanilla bean", "cocoa", "cacao", "brown sugar", "powdered sugar", "molasses",
              "maple syrup", "agave", "cornstarch", "baking powder", "baking soda", "yeast", "cream of tartar",
              "nutmeg", "cardamom", "clove", "allspice", "simple syrup", "syrup"]:
        add(n)
    # Chocolate may carry milk and soy lecithin depending on the product: UNKNOWN, never SAFE, for those allergies.
    add("chocolate", aliases=["chocolate chip", "chocolate chips", "dark chocolate"],
        possible=[A.DAIRY, A.SOY])
    add("milk chocolate", allergens=[A.DAIRY], possible=[A.SOY], traits=[T.ANIMAL_PRODUCT])
    add("white chocolate", allergens=[A.DAIRY], possible=[A.SOY], traits=[T.ANIMAL_PRODUCT])
    add("caramel", contains=["sugar"], possible=[A.DAIRY])
    add("dulce de leche", allergens=[A.DAIRY], contains=["sugar"], traits=[T.ANIMAL_PRODUCT])
    add("marshmallow", contains=["sugar", "gelatin"])
    add("whipped cream", allergens=[A.DAIRY], traits=[T.ANIMAL_PRODUCT])
    add("cream cheese", allergens=[A.DAIRY], traits=[T.ANIMAL_PRODUCT])
    add("condensed milk", aliases=["evaporated milk"], allergens=[A.DAIRY], traits=[T.ANIMAL_PRODUCT])
    add("gelato", allergens=[A.DAIRY], traits=[T.ANIMAL_PRODUCT])
    add("custard", contains=["milk", "egg", "sugar"])
    add("pudding", possible=[A.DAIRY, A.EGG])
    add("frosting", aliases=["icing"], contains=["sugar", "butter"])
    add("sorbet", contains=["sugar", "water", "fruit"])
    add("graham cracker", contains=["flour", "sugar"])
    add("pie crust", contains=["flour", "butter"])
    add("puff pastry", contains=["flour", "butter"])
    add("phyllo", aliases=["filo"], contains=["flour"])

    # --- beverage ingredients (non-alcoholic) ---
    for n in ["coffee", "espresso", "cold brew", "tea", "black tea", "green tea", "matcha", "chai"]:
        add(n, traits=[T.CAFFEINE])
    for n in ["cola", "energy drink", "yerba mate"]:
        add(n, traits=[T.CAFFEINE])
    for n in ["herbal tea", "chamomile", "rooibos", "hibiscus", "root beer", "ginger ale", "sparkling water",
              "tonic water", "club soda", "soda", "lemonade", "orange juice", "apple juice", "cranberry juice",
              "pineapple juice", "grape juice", "lime juice", "lemon juice", "coconut water", "fruit juice"]:
        add(n)
    add("oat milk", possible=[A.WHEAT_GLUTEN], contains=["oats"])  # oats are commonly cross-contaminated
    add("almond milk", allergens=[A.TREE_NUT])
    add("cashew milk", allergens=[A.TREE_NUT])
    add("soy milk", allergens=[A.SOY])
    return kg
