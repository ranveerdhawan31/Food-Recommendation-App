import unittest

from foodrec.safety import is_displayable, label_dish, safe_catalog
from foodrec.schema import (
    Allergen as A, Diet, HalalLevel, Label, MeatSource, Severity as S, UserConstraints,
)

from .fixtures import HALAL_CERTIFIED, KG, SEVERE_DAIRY, allergic, dish, rest


def label(d, user, r=None, **kw):
    return label_dish(d, r or rest(), user, KG, **kw)


class HiddenIngredients(unittest.TestCase):
    def test_casein_is_dairy(self):
        for name in ["casein", "sodium caseinate", "whey", "ghee", "paneer"]:
            with self.subTest(name):
                self.assertEqual(label(dish(ings=[name, "rice"]), SEVERE_DAIRY).label, Label.UNSAFE)

    def test_worcestershire_is_fish(self):
        user = allergic((A.FISH, S.SEVERE))
        for name in ["Worcestershire sauce", "worcestershire", "fish sauce", "caesar dressing"]:
            with self.subTest(name):
                self.assertEqual(label(dish(ings=[name]), user).label, Label.UNSAFE)

    def test_gelatin_breaks_vegetarian_and_vegan(self):
        for diet in (Diet.VEGETARIAN, Diet.VEGAN):
            res = label(dish(ings=["gelatin", "sugar"]), UserConstraints(diets={diet}))
            self.assertEqual(res.label, Label.UNSAFE)

    def test_lecithin_is_unknown_for_soy_or_egg_allergy(self):
        for allergen in (A.SOY, A.EGG):
            res = label(dish(ings=["lecithin", "sugar"]), allergic((allergen, S.SEVERE)))
            self.assertEqual(res.label, Label.UNKNOWN)
            self.assertIn(f"may_contain:{allergen.value}", res.reasons)

    def test_composites_expand_recursively(self):
        # mayonnaise -> egg, hummus -> tahini -> sesame, pesto -> pine nut + parmesan
        self.assertEqual(label(dish(ings=["mayonnaise"]), allergic((A.EGG, S.SEVERE))).label, Label.UNSAFE)
        self.assertEqual(label(dish(ings=["hummus"]), allergic((A.SESAME, S.SEVERE))).label, Label.UNSAFE)
        pesto = dish(ings=["pesto"])
        self.assertEqual(label(pesto, allergic((A.TREE_NUT, S.SEVERE))).label, Label.UNSAFE)
        self.assertEqual(label(pesto, allergic((A.DAIRY, S.SEVERE))).label, Label.UNSAFE)

    def test_soy_sauce_carries_two_allergens(self):
        res = label(dish(ings=["soy sauce"]), allergic((A.WHEAT_GLUTEN, S.SEVERE)))
        self.assertEqual(res.label, Label.UNSAFE)

    def test_honey_breaks_vegan_not_vegetarian(self):
        d = dish(ings=["honey", "oats"])
        self.assertEqual(label(d, UserConstraints(diets={Diet.VEGAN})).label, Label.UNSAFE)
        self.assertEqual(label(d, UserConstraints(diets={Diet.VEGETARIAN})).label, Label.SAFE)

    def test_clean_dish_is_safe_for_matching_user(self):
        res = label(dish(ings=["rice", "tomato", "onion"]), SEVERE_DAIRY)
        self.assertEqual(res.label, Label.SAFE)
        self.assertFalse(res.needs_verification)

    def test_entire_graph_every_allergen_node_is_unsafe_for_that_allergen(self):
        """Exhaustive: no ingredient carrying an allergen can ever be labeled SAFE."""
        for node in KG.nodes:
            for allergen in node.allergens:
                res = label(dish(ings=[node.name]), allergic((allergen, S.MILD)))
                self.assertEqual(res.label, Label.UNSAFE, f"{node.name} / {allergen}")


class UnknownHandling(unittest.TestCase):
    def test_unresolved_ingredient_is_unknown_not_safe(self):
        res = label(dish(ings=["rice", "secret house spice"]), SEVERE_DAIRY)
        self.assertEqual(res.label, Label.UNKNOWN)
        self.assertTrue(any(r.startswith("unresolved_ingredient") for r in res.reasons))

    def test_incomplete_ingredient_list_is_unknown(self):
        res = label(dish(ings=["rice"], complete=False), SEVERE_DAIRY)
        self.assertEqual(res.label, Label.UNKNOWN)
        self.assertTrue(res.needs_verification)

    def test_no_ingredient_data_is_unknown(self):
        self.assertEqual(label(dish(ings=[], complete=False), SEVERE_DAIRY).label, Label.UNKNOWN)

    def test_severe_allergy_hides_unknown_even_at_full_confidence(self):
        res = label(dish(ings=[("rice", 1.0), "mystery sauce"]), SEVERE_DAIRY)
        self.assertEqual(res.label, Label.UNKNOWN)
        self.assertFalse(is_displayable(res, SEVERE_DAIRY))

    def test_mild_allergy_shows_unknown_with_verify_flag(self):
        user = allergic((A.DAIRY, S.MILD))
        res = label(dish(ings=["rice"], complete=False), user)
        self.assertEqual(res.label, Label.UNKNOWN)
        self.assertTrue(is_displayable(res, user))
        self.assertTrue(res.needs_verification)

    def test_low_confidence_triggers_verify_label(self):
        res = label(dish(ings=[("rice", 0.6)]), allergic((A.DAIRY, S.MILD)))
        self.assertTrue(res.needs_verification)
        res_hi = label(dish(ings=[("rice", 0.95), "tomato"]), allergic((A.DAIRY, S.MILD)))
        self.assertFalse(res_hi.needs_verification)

    def test_low_confidence_detection_is_unsafe_for_severe_but_unknown_for_mild(self):
        d = dish(ings=[("butter", 0.3), "rice"], complete=False)
        self.assertEqual(label(d, allergic((A.DAIRY, S.SEVERE))).label, Label.UNSAFE)
        self.assertEqual(label(d, allergic((A.DAIRY, S.MILD))).label, Label.UNKNOWN)

    def test_incident_report_blocks_dish(self):
        d = dish("dX", ings=["rice"])
        res = label(d, SEVERE_DAIRY, incident_allergens={"dX": frozenset({A.DAIRY})})
        self.assertEqual(res.label, Label.UNSAFE)

    def test_cross_contact_warning_for_severe_only(self):
        r = rest(shared_kitchen_allergens=frozenset({A.PEANUT}))
        d = dish(ings=["rice"])
        self.assertIn("cross_contact:peanut", label(d, allergic((A.PEANUT, S.SEVERE)), r).warnings)
        self.assertEqual(label(d, allergic((A.PEANUT, S.MILD)), r).warnings, [])

    def test_unconstrained_user_sees_everything(self):
        res = label(dish(ings=["mystery"], complete=False), UserConstraints())
        self.assertEqual(res.label, Label.SAFE)


class Halal(unittest.TestCase):
    STRICT = UserConstraints(halal=HalalLevel.STRICT)
    MEAT_ONLY = UserConstraints(halal=HalalLevel.MEAT_ONLY)

    def test_pork_unsafe_everywhere(self):
        for r in (rest(**HALAL_CERTIFIED), rest(meat_source=MeatSource.HALAL)):
            for user in (self.STRICT, self.MEAT_ONLY):
                self.assertEqual(label(dish(ings=["bacon"]), user, r).label, Label.UNSAFE)

    def test_meat_needs_halal_source(self):
        d = dish(ings=["chicken", "rice"])
        self.assertEqual(label(d, self.MEAT_ONLY, rest(meat_source=MeatSource.HALAL)).label, Label.SAFE)
        self.assertEqual(label(d, self.MEAT_ONLY, rest(meat_source=MeatSource.NON_HALAL)).label, Label.UNSAFE)
        self.assertEqual(label(d, self.MEAT_ONLY, rest(meat_source=MeatSource.UNKNOWN)).label, Label.UNKNOWN)

    def test_meat_only_ignores_alcohol(self):
        d = dish(ings=["chicken", "wine"])
        r = rest(meat_source=MeatSource.HALAL)
        self.assertEqual(label(d, self.MEAT_ONLY, r).label, Label.SAFE)
        self.assertEqual(label(d, self.STRICT, r).label, Label.UNSAFE)

    def test_vegetarian_dish_ok_at_non_halal_restaurant_for_meat_only(self):
        d = dish(ings=["rice", "lentil"])
        self.assertEqual(label(d, self.MEAT_ONLY, rest(meat_source=MeatSource.NON_HALAL)).label, Label.SAFE)

    def test_incomplete_data_at_non_halal_restaurant_is_unknown(self):
        d = dish(ings=["rice"], complete=False)
        self.assertEqual(label(d, self.MEAT_ONLY, rest(meat_source=MeatSource.NON_HALAL)).label, Label.UNKNOWN)

    def test_strict_certified_kitchen_ok(self):
        d = dish(ings=["chicken", "gelatin", "rice"])
        self.assertEqual(label(d, self.STRICT, rest(**HALAL_CERTIFIED)).label, Label.SAFE)

    def test_strict_gelatin_uncertified_is_not_safe(self):
        d = dish(ings=["gelatin"])
        r = rest(halal_certified=False, meat_source=MeatSource.HALAL, alcohol_in_sauces=False)
        self.assertEqual(label(d, self.STRICT, r).label, Label.UNSAFE)

    def test_strict_alcohol_in_sauces(self):
        d = dish(ings=["chicken"])
        r_yes = rest(halal_certified=False, meat_source=MeatSource.HALAL, alcohol_in_sauces=True)
        r_unk = rest(halal_certified=False, meat_source=MeatSource.HALAL, alcohol_in_sauces=None)
        self.assertEqual(label(d, self.STRICT, r_yes).label, Label.UNSAFE)
        res = label(d, self.STRICT, r_unk)
        self.assertEqual(res.label, Label.UNKNOWN)
        self.assertFalse(is_displayable(res, self.STRICT))  # strict hides unknown

    def test_certified_kitchen_with_listed_alcohol_is_unsafe(self):
        self.assertEqual(label(dish(ings=["mirin"]), self.STRICT, rest(**HALAL_CERTIFIED)).label, Label.UNSAFE)

    def test_conflicting_restaurant_data_is_not_trusted(self):
        r = rest(halal_certified=True, meat_source=MeatSource.NON_HALAL, alcohol_in_sauces=False)
        res = label(dish(ings=["chicken"]), self.STRICT, r)
        self.assertEqual(res.label, Label.UNKNOWN)

    def test_combined_constraints_take_worst_label(self):
        user = UserConstraints(allergies={A.DAIRY: S.SEVERE}, diets={Diet.VEGETARIAN}, halal=HalalLevel.MEAT_ONLY)
        res = label(dish(ings=["rice", "butter"]), user, rest(meat_source=MeatSource.HALAL))
        self.assertEqual(res.label, Label.UNSAFE)


class CatalogFilter(unittest.TestCase):
    def test_safe_catalog_only_returns_displayable(self):
        dishes = [dish("ok", ["rice", "tomato"]), dish("bad", ["butter"]), dish("unk", ["rice"], complete=False)]
        out = [d.id for d, _ in safe_catalog(dishes, {"r1": rest()}, SEVERE_DAIRY, KG)]
        self.assertEqual(out, ["ok"])


if __name__ == "__main__":
    unittest.main()
