import random
import unittest

from foodrec.group_fusion import FairnessLedger, GroupSession, aggregate, build_member_scores
from foodrec.group_safety import Member, plan_group_order
from foodrec.mlops import (
    FeedbackEvent, IncidentRegistry, conversion_rate, default_displayable, deployment_gate,
    kg_coverage_cases, run_safety_suite,
)
from foodrec.privacy import ConsentRequired, FieldEncryptor, approximate_location
from foodrec.routing import (
    StaticProvider, UnsafeOrderError, build_group_orders, rank_quotes, verify_order_safety,
)
from foodrec.safety import label_dish
from foodrec.schema import (
    Allergen as A, Context, Label, Severity as S, SoftPreferences, UserConstraints, UserProfile,
)

from .fixtures import KG, dish, rest

CTX = Context(40.0, -74.0, 30.0, hunger=0.5, temperature_c=15.0, hour=12)


def make_session(**kw):
    pool = [dish("t1", ["rice"], cuisine="thai"), dish("t2", ["rice"], cuisine="thai", rid="r2"),
            dish("p1", ["rice"], cuisine="pizza"), dish("s1", ["rice"], cuisine="sushi", rid="r2"),
            dish("b1", ["rice"], cuisine="bbq")]
    rmap = {"r1": rest("r1"), "r2": rest("r2")}
    members = {"ann": UserProfile("ann", soft=SoftPreferences({"thai": 0.95, "pizza": 0.1, "sushi": 0.5, "bbq": 0.3})),
               "bob": UserProfile("bob", soft=SoftPreferences({"thai": 0.2, "pizza": 0.9, "sushi": 0.5, "bbq": 0.8}))}
    scores = build_member_scores(members, pool, rmap, CTX)
    return GroupSession(list(members), pool, scores, rng=random.Random(3), **kw), pool


class Fusion(unittest.TestCase):
    def test_least_misery_and_nash(self):
        s = {"a": 0.9, "b": 0.1}
        self.assertAlmostEqual(aggregate(s, "least_misery"), 0.1)
        self.assertLess(aggregate(s, "nash"), 0.5)
        self.assertGreater(aggregate({"a": 0.6, "b": 0.6}, "nash"), aggregate(s, "nash"))

    def test_fairness_weight_favors_past_loser(self):
        ledger = FairnessLedger(strength=2.0)
        ledger.record({"a": 1.0, "b": 0.1}, {"a": 1.0, "b": 1.0})  # b lost last time
        w = ledger.weights(["a", "b"])
        self.assertGreater(w["b"], w["a"])
        # dish liked by b but not a now scores relatively higher than before the penalty
        for method in ("nash", "least_misery"):
            fav_b = aggregate({"a": 0.3, "b": 0.9}, method, w)
            fav_a = aggregate({"a": 0.9, "b": 0.3}, method, w)
            self.assertGreater(fav_b, fav_a)

    def test_shuffle_returns_distinct_options_with_per_person_match(self):
        sess, _ = make_session()
        first = sess.shuffle()
        self.assertEqual(len(first), 3)
        self.assertEqual(len({o["dish_id"] for o in first}), 3)
        self.assertEqual(set(first[0]["match"]), {"ann", "bob"})
        second = sess.shuffle()  # reroll excludes already-shown while fresh dishes remain
        self.assertTrue(len(second) >= 1)

    def test_votes_shift_member_weights_and_reach_consensus(self):
        sess, pool = make_session()
        sushi = pool[3]  # mid-range for both members, so a vote can move the score
        before = sess.per_person_match(sushi)["bob"]
        sess.vote("bob", "s1", like=False)
        self.assertLess(sess.per_person_match(sushi)["bob"], before)
        sess.vote("ann", "s1", like=True)
        self.assertIsNone(sess.consensus())  # bob disliked it
        sess.vote("bob", "p1", True)
        sess.vote("ann", "p1", True)
        self.assertEqual(sess.consensus(), "p1")

    def test_session_rejects_dishes_outside_the_safe_pool(self):
        sess, _ = make_session()
        out = sess.handle_event({"type": "vote", "member": "ann", "dish_id": "UNSAFE_INJECTED", "like": True})
        self.assertEqual(out["type"], "error")
        out = sess.handle_event({"type": "finalize", "dish_id": "UNSAFE_INJECTED"})
        self.assertEqual(out["type"], "error")
        self.assertEqual(sess.handle_event({"type": "vote", "member": "eve", "dish_id": "t1", "like": True})["type"], "error")

    def test_finalize_updates_ledger(self):
        sess, _ = make_session()
        sess.handle_event({"type": "finalize", "dish_id": "t1"})  # ann's favorite, bob's least
        w = sess.ledger.weights(["ann", "bob"])
        self.assertGreater(w["bob"], w["ann"])

    def test_event_protocol(self):
        sess, _ = make_session()
        self.assertEqual(sess.handle_event({"type": "reroll"})["type"], "options")
        self.assertEqual(sess.handle_event({"type": "nope"})["type"], "error")

    def test_end_to_end_group_plan_feeds_session_only_safe_dishes(self):
        members = [Member("ann", UserConstraints(allergies={A.DAIRY: S.SEVERE})),
                   Member("bob", UserConstraints(allergies={A.EGG: S.SEVERE}))]
        rests = {"r1": rest("r1")}
        ds = [dish("ok1", ["rice"]), dish("ok2", ["tomato"]), dish("d", ["butter"]), dish("e", ["egg"])]
        plan = plan_group_order(members, rests, ds, KG)
        self.assertEqual(plan.mode, "SHARED")
        profiles = {m.id: UserProfile(m.id) for m in members}
        scores = build_member_scores(profiles, plan.shared_pool, rests, CTX)
        sess = GroupSession(["ann", "bob"], plan.shared_pool, scores, rng=random.Random(0), page_size=2)
        shown = {o["dish_id"] for o in sess.shuffle()}
        self.assertEqual(shown, {"ok1", "ok2"})


class Routing(unittest.TestCase):
    def setUp(self):
        self.cheap = StaticProvider("cheapo", {"r1": {"prices": {"d1": 10.0, "d2": 12.0}, "delivery_fee": 1.0,
                                                      "service_fee": 1.0, "eta_min": 50}})
        self.fast = StaticProvider("fasty", {"r1": {"prices": {"d1": 10.0, "d2": 12.0}, "delivery_fee": 3.0,
                                                    "service_fee": 2.0, "eta_min": 20}})
        self.templates = {"cheapo": "https://example.test/c/{restaurant_id}?items={items}",
                          "fasty": "https://example.test/f/{restaurant_id}?items={items}"}

    def test_rank_by_cost_plus_time_and_deep_link(self):
        # cheapo: 12 total + 50 min; fasty: 15 total + 20 min
        slow_ok = rank_quotes([self.cheap, self.fast], "r1", ["d1"], (0, 0), self.templates, value_of_time=0.0)
        self.assertEqual([q.provider for q in slow_ok], ["cheapo", "fasty"])  # time is free -> cheapest wins
        hurry = rank_quotes([self.cheap, self.fast], "r1", ["d1"], (0, 0), self.templates, value_of_time=0.2)
        self.assertEqual([q.provider for q in hurry], ["fasty", "cheapo"])  # 15+4 < 12+10
        self.assertTrue(hurry[0].deep_link.startswith("https://example.test/f/r1?items=d1"))

    def test_unavailable_items_are_dropped(self):
        qs = rank_quotes([self.cheap], "r1", ["d1", "nope"], (0, 0))
        self.assertEqual(qs, [])

    def test_missing_template_gives_no_link(self):
        qs = rank_quotes([self.cheap], "r1", ["d1"], (0, 0), {})
        self.assertIsNone(qs[0].deep_link)

    def test_order_time_safety_recheck_blocks_changed_menu(self):
        member = Member("ann", UserConstraints(allergies={A.DAIRY: S.SEVERE}))
        recipe_changed = dish("d1", ["rice", "butter"])  # menu now lists butter
        rmap = {"r1": rest("r1")}
        plan = plan_group_order([Member("ann", member.constraints)], rmap, [dish("d1", ["rice"])], KG)
        with self.assertRaises(UnsafeOrderError):
            build_group_orders({"ann": [recipe_changed]}, [member], plan, rmap, KG, [self.cheap], (0, 0))

    def test_group_cart_vs_split(self):
        members = [Member("ann", UserConstraints()), Member("bob", UserConstraints())]
        rmap = {"r1": rest("r1")}
        picks = {"ann": [dish("d1", ["rice"])], "bob": [dish("d2", ["tomato"])]}
        plan = plan_group_order(members, rmap, [d for ds in picks.values() for d in ds], KG)
        one = build_group_orders(picks, members, plan, rmap, KG, [self.cheap], (0, 0))
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0].quotes[0].item_total, 22.0)
        two = build_group_orders(picks, members, plan, rmap, KG, [self.cheap], (0, 0), split=True)
        self.assertEqual(len(two), 2)


class MLOps(unittest.TestCase):
    def test_kg_coverage_suite_has_zero_violations(self):
        cases = kg_coverage_cases(KG, rest())
        self.assertGreater(len(cases), 50)
        self.assertEqual(run_safety_suite(default_displayable(KG), cases), [])

    def test_gate_catches_a_broken_pipeline(self):
        cases = kg_coverage_cases(KG, rest())
        broken = lambda dish_, rest_, constraints: True  # shows everything
        violations = run_safety_suite(broken, cases)
        self.assertGreater(len(violations), 0)
        report = deployment_gate(violations, 0.10, 0.20, 30.0, 20.0)  # better business metrics, still blocked
        self.assertFalse(report.passed)
        self.assertTrue(report.reasons[0].startswith("safety violations"))

    def test_gate_passes_clean_candidate_and_flags_regressions(self):
        self.assertTrue(deployment_gate([], 0.10, 0.11, 30.0, 31.0).passed)
        self.assertFalse(deployment_gate([], 0.10, 0.05, 30.0, 30.0).passed)
        self.assertFalse(deployment_gate([], 0.10, 0.10, 30.0, 40.0).passed)

    def test_incident_report_quarantines_dish_immediately(self):
        reg = IncidentRegistry()
        d = dish("d1", ["rice"])
        user = UserConstraints(allergies={A.SESAME: S.SEVERE})
        self.assertEqual(label_dish(d, rest(), user, KG, incident_allergens=reg.as_mapping()).label, Label.SAFE)
        reg.record(FeedbackEvent("u2", "d1", "allergy_incident", allergen=A.SESAME))
        self.assertEqual(label_dish(d, rest(), user, KG, incident_allergens=reg.as_mapping()).label, Label.UNSAFE)
        with self.assertRaises(ValueError):
            reg.record(FeedbackEvent("u", "d1", "click"))

    def test_conversion_rate(self):
        ev = [FeedbackEvent("u", "d", "view")] * 4 + [FeedbackEvent("u", "d", "order")]
        self.assertAlmostEqual(conversion_rate(ev), 0.25)


class Privacy(unittest.TestCase):
    def test_approximate_location(self):
        self.assertEqual(approximate_location(40.712776, -74.005974), (40.71, -74.01))

    def test_encryption_roundtrip_requires_consent(self):
        enc = FieldEncryptor(FieldEncryptor.generate_key())
        profile = UserProfile("u", UserConstraints(allergies={A.PEANUT: S.SEVERE}))
        with self.assertRaises(ConsentRequired):
            enc.encrypt_constraints(profile)
        profile.consent_given = True
        token = enc.encrypt_constraints(profile)
        self.assertNotIn(b"peanut", token)
        self.assertEqual(enc.decrypt_constraints(token).allergies, {A.PEANUT: S.SEVERE})


if __name__ == "__main__":
    unittest.main()
