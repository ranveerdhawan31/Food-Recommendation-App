import random
import unittest

from foodrec.pipeline import recommend, recommend_meal
from foodrec.ranking import FEATURE_NAMES, HeuristicRanker, assert_no_safety_features, build_features
from foodrec.safety import is_displayable, label_dish, verification_warning
from foodrec.schema import (
    Allergen as A, Context, Diet, HalalLevel, Label, Severity as S, SoftPreferences, UserConstraints, UserProfile,
)
from foodrec.survey import NA_CUISINES, SoftSurvey, build_taste_preferences
from foodrec.taste import taste_filter

from .fixtures import KG, SEVERE_DAIRY, allergic, dish, rest

CTX = Context(lat=40.0, lon=-74.0, budget=30.0, hunger=0.5, temperature_c=20.0, hour=12)
RMAP = {"r1": rest()}
NONE = UserConstraints()


def label(d, c):
    return label_dish(d, rest(), c, KG)


def drink(did="tea", ings=("tea", "water"), alcohol_free=True, **kw):
    return dish(did, list(ings), category="beverage", alcohol_free=alcohol_free, **kw)


class VerificationWarning(unittest.TestCase):
    def test_unknown_ingredient_warns_even_without_constraints(self):
        res = label(dish(ings=["rice", "secret house spice"]), NONE)
        self.assertEqual(res.label, Label.SAFE)  # nothing hard is at stake
        self.assertTrue(res.needs_verification)
        text = verification_warning(res, NONE)
        self.assertIn("secret house spice", text)
        self.assertTrue(text.startswith("Verify with the restaurant"))
        self.assertNotIn("medical guarantee", text)

    def test_missing_ingredient_data_warns(self):
        res = label(dish(ings=[], complete=False), NONE)
        self.assertIn("No ingredient information", verification_warning(res, NONE))

    def test_fully_known_dish_has_no_warning(self):
        res = label(dish(ings=["rice", "tomato"]), NONE)
        self.assertIsNone(verification_warning(res, NONE))

    def test_incomplete_but_identified_list_does_not_nag_unconstrained_user(self):
        self.assertFalse(label(dish(ings=["rice"], complete=False), NONE).needs_verification)

    def test_constrained_user_gets_disclaimer_and_incomplete_notice(self):
        user = allergic((A.DAIRY, S.MILD))
        res = label(dish(ings=["rice"], complete=False), user)
        text = verification_warning(res, user)
        self.assertIn("incomplete", text)
        self.assertIn("medical guarantee", text)

    def test_long_unknown_lists_are_truncated(self):
        res = label(dish(ings=[f"mystery{i}" for i in range(9)]), NONE)
        text = verification_warning(res, NONE)
        self.assertIn("and 4 more", text)

    def test_recommendation_carries_warning(self):
        user = UserProfile("u")
        ds = [dish("a", ["rice", "secret house spice"], cuisine="x"), dish("b", ["rice"], cuisine="x")]
        out = {r.dish.id: r for r in recommend(user, CTX, ds, RMAP, KG, n=2, explore_sigma=0.0)}
        self.assertIn("secret house spice", out["a"].warning)
        self.assertIsNone(out["b"].warning)


class BeverageAlcoholPolicy(unittest.TestCase):
    def test_unconfirmed_alcohol_status_fails_closed_for_everyone(self):
        res = label(drink(alcohol_free=None), NONE)
        self.assertEqual(res.label, Label.UNSAFE)
        self.assertFalse(is_displayable(res, NONE))
        self.assertEqual(label(drink(alcohol_free=False), NONE).label, Label.UNSAFE)

    def test_alcohol_ingredient_overrides_provider_flag(self):
        res = label(drink("mock", ["lime juice", "rum"], alcohol_free=True), NONE)
        self.assertEqual(res.label, Label.UNSAFE)
        self.assertIn("beverage:contains_alcohol", res.reasons)

    def test_confirmed_clean_drink_is_shown(self):
        self.assertTrue(is_displayable(label(drink(), NONE), NONE))

    def test_root_beer_is_not_mistaken_for_beer(self):
        self.assertEqual(label(drink("rb", ["root beer"]), NONE).label, Label.SAFE)

    def test_kombucha_and_vanilla_extract_are_treated_as_alcohol(self):
        self.assertEqual(label(drink("k", ["kombucha"]), NONE).label, Label.UNSAFE)
        self.assertEqual(label(drink("v", ["milk", "vanilla extract"]), NONE).label, Label.UNSAFE)

    def test_non_beverages_are_unaffected(self):
        self.assertEqual(label(dish(ings=["rice"], alcohol_free=None), NONE).label, Label.SAFE)

    def test_allergy_still_applies_to_drinks(self):
        latte = drink("latte", ["espresso", "milk"])
        self.assertEqual(label(latte, SEVERE_DAIRY).label, Label.UNSAFE)
        oat = drink("oat", ["oat milk", "espresso"])
        self.assertEqual(label(oat, allergic((A.WHEAT_GLUTEN, S.SEVERE))).label, Label.UNKNOWN)


class DessertSafety(unittest.TestCase):
    def test_chocolate_is_unknown_not_safe_for_dairy_and_soy_allergies(self):
        d = dish("brownie", ["chocolate", "sugar", "flour"], category="dessert")
        self.assertEqual(label(d, allergic((A.DAIRY, S.SEVERE))).label, Label.UNKNOWN)
        self.assertEqual(label(d, allergic((A.SOY, S.SEVERE))).label, Label.UNKNOWN)

    def test_marshmallow_breaks_vegetarian_via_gelatin(self):
        d = dish("smore", ["marshmallow", "chocolate"], category="dessert")
        self.assertEqual(label(d, UserConstraints(diets={Diet.VEGETARIAN})).label, Label.UNSAFE)

    def test_strict_halal_blocks_vanilla_extract_dessert(self):
        d = dish("cake", ["flour", "sugar", "vanilla extract"], category="dessert")
        self.assertEqual(label(d, UserConstraints(halal=HalalLevel.STRICT)).label, Label.UNSAFE)


class CuisinePreferences(unittest.TestCase):
    def test_na_cuisine_list_has_no_duplicates(self):
        self.assertEqual(len(NA_CUISINES), len(set(NA_CUISINES)))

    def test_declared_cuisines_are_not_reasked_and_shift_beliefs(self):
        sv = SoftSurvey(["thai", "pizza", "sushi"], max_questions=3)
        sv.declare(liked=("thai",), avoided=("sushi",))
        asked = []
        while (q := sv.next_question()) is not None:
            asked.append(q)
            sv.answer(q, True)
        self.assertEqual(asked, ["pizza"])
        soft = sv.to_preferences()
        self.assertGreater(soft.cuisines["thai"], 0.8)
        self.assertLess(soft.cuisines["sushi"], 0.2)
        self.assertEqual(soft.cuisines_avoided, frozenset({"sushi"}))

    def test_declare_rejects_typos_and_contradictions(self):
        sv = SoftSurvey(["thai", "pizza"])
        with self.assertRaises(ValueError):
            sv.declare(liked=("thia",))
        with self.assertRaises(ValueError):
            sv.declare(liked=("thai",), avoided=("thai",))

    def test_avoided_cuisine_never_returned_even_with_huge_noise(self):
        soft = SoftPreferences({"thai": 0.1, "pizza": 0.9}, cuisines_avoided=frozenset({"pizza"}))
        user = UserProfile("u", soft=soft)
        ds = [dish("p1", ["rice"], cuisine="pizza"), dish("t1", ["rice"], cuisine="thai")]
        for seed in range(30):
            out = recommend(user, CTX, ds, RMAP, KG, rng=random.Random(seed), explore_sigma=50.0)
            self.assertEqual([r.dish.id for r in out], ["t1"])

    def test_liked_cuisine_ranks_first(self):
        user = UserProfile("u", soft=SoftPreferences({"thai": 0.95, "pizza": 0.1}))
        ds = [dish("p1", ["rice"], cuisine="pizza"), dish("t1", ["rice"], cuisine="thai")]
        out = recommend(user, CTX, ds, RMAP, KG, n=2, explore_sigma=0.0)
        self.assertEqual(out[0].dish.id, "t1")


class SweetCravings(unittest.TestCase):
    def setUp(self):
        self.mild = dish("mild", ["sugar"], category="dessert", sweetness=1, cuisine="x")
        self.rich = dish("rich", ["sugar"], category="dessert", sweetness=5, cuisine="x")
        self.ranker = HeuristicRanker()

    def top(self, soft, ctx):
        rows = [build_features(d, rest(), soft, ctx, 0.0) for d in (self.mild, self.rich)]
        scores = self.ranker.score(rows)
        return ("mild", "rich")[scores.index(max(scores))]

    def test_strong_craving_prefers_sweeter_dessert(self):
        ctx = Context(40.0, -74.0, 30.0, sweet_craving=1.0)
        self.assertEqual(self.top(SoftPreferences({"x": 0.5}, sweet_tooth=0.2), ctx), "rich")

    def test_no_craving_and_low_sweet_tooth_prefers_milder(self):
        ctx = Context(40.0, -74.0, 30.0)
        self.assertEqual(self.top(SoftPreferences({"x": 0.5}, sweet_tooth=0.1), ctx), "mild")

    def test_sweetness_is_neutral_for_mains(self):
        main = dish("m", ["rice"], sweetness=5, cuisine="x")
        f = build_features(main, rest(), SoftPreferences(), Context(40.0, -74.0, 30.0, sweet_craving=1.0), 0.0)
        self.assertEqual(f["sweetness_fit"], 0.0)

    def test_tag_affinity_orders_desserts(self):
        soft = SoftPreferences({"x": 0.5}, dessert_tags={"chocolate": 0.95, "fruity": 0.05})
        choc = dish("c", ["sugar"], category="dessert", tags=("chocolate",), sweetness=3, cuisine="x")
        fruit = dish("f", ["sugar"], category="dessert", tags=("fruity",), sweetness=3, cuisine="x")
        user = UserProfile("u", soft=soft)
        out = recommend(user, CTX, [choc, fruit], RMAP, KG, category="dessert", n=2, explore_sigma=0.0)
        self.assertEqual(out[0].dish.id, "c")

    def test_features_still_free_of_safety_terms(self):
        assert_no_safety_features(FEATURE_NAMES)
        self.assertIn("sweetness_fit", FEATURE_NAMES)


class CategoriesAndCaffeine(unittest.TestCase):
    def test_default_category_is_main_only(self):
        ds = [dish("m", ["rice"], cuisine="x"), dish("d", ["sugar"], category="dessert", cuisine="x")]
        out = recommend(UserProfile("u"), CTX, ds, RMAP, KG, n=5, explore_sigma=0.0)
        self.assertEqual([r.dish.id for r in out], ["m"])
        out_all = recommend(UserProfile("u"), CTX, ds, RMAP, KG, n=5, explore_sigma=0.0, category=None)
        self.assertEqual({r.dish.id for r in out_all}, {"m", "d"})

    def test_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            taste_filter([], SoftPreferences(), KG, "snack")

    def test_caffeine_free_hides_known_caffeine_only(self):
        soft = SoftPreferences(caffeine_ok=False)
        coffee = drink("coffee", ["espresso", "water"])
        flagged = drink("flag", ["water"], caffeinated=True)
        herbal = drink("herbal", ["herbal tea"])
        unknown = drink("unk", ["water", "mystery syrup"])
        kept = [d.id for d in taste_filter([coffee, flagged, herbal, unknown], soft, KG, "beverage")]
        self.assertEqual(kept, ["herbal", "unk"])

    def test_build_taste_preferences_validates(self):
        base = SoftPreferences({"thai": 0.7}, cuisines_avoided=frozenset({"sushi"}))
        p = build_taste_preferences(base, sweet_tooth=0.8, dessert_likes=("chocolate",),
                                    beverage_dislikes=("soda",), caffeine_ok=False)
        self.assertEqual(p.cuisines_avoided, frozenset({"sushi"}))
        self.assertEqual(p.dessert_tags["chocolate"], 0.9)
        self.assertEqual(p.beverage_tags["soda"], 0.1)
        self.assertFalse(p.caffeine_ok)
        with self.assertRaises(ValueError):
            build_taste_preferences(base, dessert_likes=("choclate",))
        with self.assertRaises(ValueError):
            build_taste_preferences(base, sweet_tooth=1.5)


class MealSafetyInvariants(unittest.TestCase):
    def test_recommend_meal_returns_three_categories(self):
        ds = [dish("m", ["rice"], cuisine="x"), dish("d", ["sugar"], category="dessert", cuisine="x"),
              drink("t", ["tea", "water"], cuisine="x")]
        meal = recommend_meal(UserProfile("u"), CTX, ds, RMAP, KG, explore_sigma=0.0)
        self.assertEqual([r.dish.id for r in meal.mains], ["m"])
        self.assertEqual([r.dish.id for r in meal.desserts], ["d"])
        self.assertEqual([r.dish.id for r in meal.beverages], ["t"])

    def test_no_sweet_interest_skips_desserts(self):
        ds = [dish("d", ["sugar"], category="dessert", cuisine="x")]
        user = UserProfile("u", soft=SoftPreferences(sweet_tooth=0.0))
        self.assertEqual(recommend_meal(user, CTX, ds, RMAP, KG).desserts, [])
        craving = Context(40.0, -74.0, 30.0, sweet_craving=0.8)
        self.assertEqual(len(recommend_meal(user, craving, ds, RMAP, KG).desserts), 1)

    def test_randomized_every_item_passes_the_users_own_constraints(self):
        rng = random.Random(7)
        pool = ["rice", "sugar", "milk", "butter", "chocolate", "egg", "flour", "tea", "espresso", "oat milk",
                "almond milk", "soy milk", "marshmallow", "rum", "vanilla extract", "kombucha", "mystery goo",
                "peanut butter", "lime juice", "honey", "gelatin", "pork", "wine"]
        for trial in range(200):
            constraints = UserConstraints(
                allergies={a: rng.choice(list(S)) for a in rng.sample(list(A), rng.randint(0, 3))},
                diets={d for d in Diet if rng.random() < 0.25},
                halal=rng.choice(list(HalalLevel)),
            )
            ds = []
            for i in range(30):
                cat = rng.choice(["main", "dessert", "beverage"])
                ds.append(dish(f"d{i}", rng.sample(pool, rng.randint(1, 4)), complete=rng.random() < 0.8,
                               category=cat, alcohol_free=rng.choice([True, True, False, None]),
                               sweetness=rng.randint(0, 5), cuisine=rng.choice(["x", "y"])))
            user = UserProfile("u", constraints, SoftPreferences(caffeine_ok=rng.random() < 0.5))
            meal = recommend_meal(user, CTX, ds, RMAP, KG, rng=random.Random(trial), explore_sigma=5.0)
            for r in meal.mains + meal.desserts + meal.beverages:
                res = label_dish(r.dish, RMAP["r1"], constraints, KG)
                self.assertTrue(is_displayable(res, constraints), f"trial {trial}: {r.dish.id} {res.reasons}")
                if r.dish.category == "beverage":
                    self.assertIs(r.dish.alcohol_free, True)
                    self.assertNotIn("beverage:contains_alcohol", res.reasons)
                # the warning must be present whenever verification is needed
                self.assertEqual(r.warning is not None, res.needs_verification)


if __name__ == "__main__":
    unittest.main()
