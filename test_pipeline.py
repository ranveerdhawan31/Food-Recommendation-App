import random
import unittest

from foodrec.inference import RuleBasedExtractor, enrich_dish
from foodrec.pipeline import recommend
from foodrec.ranking import (
    FEATURE_NAMES, HeuristicRanker, assert_no_safety_features, build_features, mmr,
)
from foodrec.retrieval import is_open, operational_filter
from foodrec.safety import label_dish
from foodrec.schema import (
    Allergen as A, Context, Dish, Label, Severity as S, SoftPreferences, UserConstraints, UserProfile,
)
from foodrec.survey import SoftSurvey, bernoulli_entropy, build_constraints

from .fixtures import KG, dish, rest

CTX = Context(lat=40.0, lon=-74.0, budget=20.0, hunger=0.6, temperature_c=5.0, hour=12)


class Survey(unittest.TestCase):
    def test_hard_constraints_built_only_from_answers(self):
        c = build_constraints({"peanut": "severe", "dairy": "mild"}, ["vegetarian"], "strict")
        self.assertEqual(c.allergies[A.PEANUT], S.SEVERE)
        self.assertTrue(c.exclude_unknown)

    def test_most_uncertain_cuisine_is_asked_first_and_stops(self):
        sv = SoftSurvey(["thai", "pizza", "sushi"], priors={"thai": 0.5, "pizza": 0.9, "sushi": 0.1}, max_questions=2)
        self.assertEqual(sv.next_question(), "thai")
        sv.answer("thai", True)
        self.assertIn(sv.next_question(), ("pizza", "sushi", None))
        self.assertGreater(sv.belief("thai"), 0.5)
        sv.answer("pizza", False)
        self.assertIsNone(sv.next_question())  # max_questions reached

    def test_entropy_peaks_at_half(self):
        self.assertGreater(bernoulli_entropy(0.5), bernoulli_entropy(0.9))
        self.assertEqual(bernoulli_entropy(1.0), 0.0)


class Inference(unittest.TestCase):
    def test_longest_match_and_never_complete(self):
        ex = RuleBasedExtractor(KG)
        d = Dish("x", "r1", "Satay skewers", 9.0, description="chicken with peanut butter and rice")
        enrich_dish(d, ex)
        names = {m.text for m in d.ingredients}
        self.assertIn("peanut butter", names)
        self.assertNotIn("butter", names)  # consumed by the longer match
        self.assertFalse(d.ingredients_complete)
        self.assertTrue(all(m.source == "nlp" for m in d.ingredients))

    def test_dish_name_priors_have_lower_confidence(self):
        d = Dish("c", "r1", "Chicken Caesar Salad", 12.0)
        enrich_dish(d, RuleBasedExtractor(KG))
        conf = {m.text: m.confidence for m in d.ingredients}
        self.assertLess(conf["caesar dressing"], 0.9)

    def test_inferred_dish_is_never_safe_for_severe_allergy(self):
        d = Dish("c", "r1", "Garden salad", 8.0, description="lettuce, tomato, cucumber")
        enrich_dish(d, RuleBasedExtractor(KG))
        res = label_dish(d, rest(), UserConstraints(allergies={A.DAIRY: S.SEVERE}), KG)
        self.assertEqual(res.label, Label.UNKNOWN)


class Retrieval(unittest.TestCase):
    def test_operational_filters(self):
        near, far, closed = rest("near"), rest("far", lat=41.0), rest("closed", open_hour=18, close_hour=23)
        rmap = {r.id: r for r in (near, far, closed)}
        ds = [dish("a", rid="near"), dish("b", rid="far"), dish("c", rid="closed"), dish("exp", rid="near", price=99)]
        out = [d.id for d in operational_filter(ds, rmap, CTX)]
        self.assertEqual(out, ["a"])

    def test_overnight_hours(self):
        r = rest(open_hour=22, close_hour=4)
        self.assertTrue(is_open(r, 23) and is_open(r, 2))
        self.assertFalse(is_open(r, 12))


class Ranking(unittest.TestCase):
    def test_no_safety_attribute_is_a_feature(self):
        assert_no_safety_features(FEATURE_NAMES)
        with self.assertRaises(ValueError):
            assert_no_safety_features(["price_ratio", "halal_flag"])
        with self.assertRaises(ValueError):
            HeuristicRanker({"allergen_score": 1.0})

    def test_weather_and_hunger_shift_scores(self):
        soup = dish("soup", is_hot=True, portion=1.2, cuisine="x")
        salad = dish("salad", is_hot=False, portion=1.2, cuisine="x")
        soft, r = SoftPreferences({"x": 0.5}), rest()
        ranker = HeuristicRanker()
        cold = Context(40.0, -74.0, 20.0, hunger=0.7, temperature_c=2.0)
        hot = Context(40.0, -74.0, 20.0, hunger=0.7, temperature_c=35.0)
        s_cold = ranker.score([build_features(soup, r, soft, cold, 0.0), build_features(salad, r, soft, cold, 0.0)])
        s_hot = ranker.score([build_features(soup, r, soft, hot, 0.0), build_features(salad, r, soft, hot, 0.0)])
        self.assertGreater(s_cold[0], s_cold[1])
        self.assertGreater(s_hot[1], s_hot[0])

    def test_mmr_diversifies(self):
        ds = [dish("a1", cuisine="thai"), dish("a2", cuisine="thai"), dish("b1", cuisine="pizza")]
        order = mmr(ds, [1.0, 0.95, 0.9], k=2, lam=0.5)
        self.assertEqual([ds[i].id for i in order], ["a1", "b1"])


class PipelineSafetyOverride(unittest.TestCase):
    def test_ml_score_cannot_resurface_blocked_dish(self):
        class FavoriteIsUnsafe:  # adversarial ranker: loves anything with butter
            def score(self, rows):
                return [100.0 if i == 0 else 0.0 for i, _ in enumerate(rows)]

        user = UserProfile("u", UserConstraints(allergies={A.DAIRY: S.SEVERE}),
                           SoftPreferences({"french": 1.0}), consent_given=True)
        rmap = {"r1": rest()}
        ds = [dish("butter_dish", ["butter"], cuisine="french"), dish("rice", ["rice"], cuisine="french"),
              dish("mystery", ["rice"], complete=False, cuisine="french")]
        for seed in range(25):
            out = recommend(user, CTX, ds, rmap, KG, ranker=FavoriteIsUnsafe(), rng=random.Random(seed),
                            explore_sigma=5.0)
            self.assertEqual([r.dish.id for r in out], ["rice"])

    def test_empty_when_everything_unsafe(self):
        user = UserProfile("u", UserConstraints(allergies={A.DAIRY: S.SEVERE}))
        out = recommend(user, CTX, [dish("b", ["butter"])], {"r1": rest()}, KG)
        self.assertEqual(out, [])

    def test_returns_ranked_diverse_results(self):
        user = UserProfile("u", soft=SoftPreferences({"thai": 0.9, "pizza": 0.2}))
        ds = [dish("t1", ["rice"], cuisine="thai"), dish("t2", ["rice"], cuisine="thai"),
              dish("p1", ["rice"], cuisine="pizza")]
        out = recommend(user, CTX, ds, {"r1": rest()}, KG, n=3, rng=random.Random(1), explore_sigma=0.0)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].dish.cuisine, "thai")


if __name__ == "__main__":
    unittest.main()
