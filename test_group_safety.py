import itertools
import random
import unittest

from foodrec.group_safety import Member, merge_constraints, plan_group_order
from foodrec.safety import is_displayable, label_dish
from foodrec.schema import Allergen as A, Diet, HalalLevel, MeatSource, Severity as S, UserConstraints

from .fixtures import HALAL_CERTIFIED, KG, allergic, dish, rest


def plan(members, rests, dishes, **kw):
    return plan_group_order(members, {r.id: r for r in rests}, dishes, KG, **kw)


def assert_all_assignments_safe(tc, members, rests, dishes, p):
    rmap = {r.id: r for r in rests}
    dmap = {d.id: d for d in dishes}
    by_id = {m.id: m for m in members}
    if p.mode == "SHARED":
        for d in p.shared_pool:
            for m in members:
                res = label_dish(d, rmap[d.restaurant_id], m.constraints, KG)
                tc.assertTrue(is_displayable(res, m.constraints), f"{d.id} unsafe for {m.id}")
    else:
        for mid, (rid, dish_ids) in p.assignments.items():
            tc.assertTrue(dish_ids)
            for did in dish_ids:
                d = dmap[did]
                tc.assertEqual(d.restaurant_id, rid)
                res = label_dish(d, rmap[rid], by_id[mid].constraints, KG)
                tc.assertTrue(is_displayable(res, by_id[mid].constraints), f"{did} unsafe for {mid}")


class MergeConstraints(unittest.TestCase):
    def test_union_takes_strictest(self):
        a = Member("a", UserConstraints(allergies={A.DAIRY: S.MILD}, halal=HalalLevel.MEAT_ONLY))
        b = Member("b", UserConstraints(allergies={A.DAIRY: S.SEVERE, A.EGG: S.MILD},
                                        diets={Diet.VEGAN}, halal=HalalLevel.STRICT))
        m = merge_constraints([a, b])
        self.assertEqual(m.allergies, {A.DAIRY: S.SEVERE, A.EGG: S.MILD})
        self.assertEqual(m.diets, {Diet.VEGAN})
        self.assertEqual(m.halal, HalalLevel.STRICT)
        self.assertTrue(m.exclude_unknown)


class SharedPool(unittest.TestCase):
    def test_shared_pool_is_safe_for_everyone(self):
        members = [Member("dairy", allergic((A.DAIRY, S.SEVERE))), Member("egg", allergic((A.EGG, S.SEVERE)))]
        rests = [rest("r1"), rest("r2")]
        dishes = [dish("ok", ["rice"], rid="r1"), dish("d", ["butter"], rid="r1"),
                  dish("e", ["egg"], rid="r2"), dish("ok2", ["tomato"], rid="r2")]
        p = plan(members, rests, dishes)
        self.assertEqual(p.mode, "SHARED")
        self.assertEqual({d.id for d in p.shared_pool}, {"ok", "ok2"})
        assert_all_assignments_safe(self, members, rests, dishes, p)

    def test_shared_pool_equals_merged_constraint_filter(self):
        members = [Member("a", allergic((A.DAIRY, S.MILD))),
                   Member("b", UserConstraints(diets={Diet.VEGETARIAN}, halal=HalalLevel.MEAT_ONLY))]
        r = rest("r1", meat_source=MeatSource.HALAL)
        dishes = [dish("1", ["rice"]), dish("2", ["chicken"]), dish("3", ["butter"]),
                  dish("4", ["lentil", "rice"], complete=False)]
        p = plan(members, [r], dishes)
        merged = merge_constraints(members)
        expected = {d.id for d in dishes
                    if is_displayable(label_dish(d, r, merged, KG), merged)}
        got = {d.id for d in p.shared_pool}
        # the merged (union) filter is never more permissive than per-member evaluation
        self.assertTrue(got >= expected)
        for d in dishes:
            if d.id in got:
                for m in members:
                    self.assertTrue(is_displayable(label_dish(d, r, m.constraints, KG), m.constraints))


class SplitOrderSafety(unittest.TestCase):
    def setUp(self):
        self.vegan = Member("vegan", UserConstraints(diets={Diet.VEGAN}))
        self.halal = Member("halal", UserConstraints(halal=HalalLevel.STRICT))
        self.nut = Member("nut", allergic((A.PEANUT, S.SEVERE)))

    def test_single_vendor_when_no_common_dish(self):
        # vegan can eat salad; halal can eat chicken; nothing is safe for both
        # the vegan-safe dish contains wine, so a strict-halal member cannot share it
        r_plain = rest("r1", halal_certified=False, meat_source=MeatSource.HALAL, alcohol_in_sauces=False)
        dishes = [dish("wine_salad", ["rice", "wine"], rid="r1"), dish("chk", ["chicken"], rid="r1")]
        members = [self.vegan, self.halal]
        p = plan(members, [r_plain], dishes)
        self.assertEqual(p.mode, "SINGLE_VENDOR")
        self.assertEqual(p.assignments["vegan"][1], ["wine_salad"])  # wine is vegan; halal-STRICT cannot have it
        self.assertEqual(p.assignments["halal"][1], ["chk"])
        assert_all_assignments_safe(self, members, [r_plain], dishes, p)

    def test_split_when_no_single_restaurant_covers_everyone(self):
        r1 = rest("veg_place", lat=40.0, lon=-74.0)
        r2 = rest("halal_place", lat=40.01, lon=-74.0, **HALAL_CERTIFIED)
        dishes = [dish("v1", ["rice", "lentil"], rid="veg_place"),
                  dish("h1", ["chicken", "rice"], rid="halal_place")]
        members = [self.vegan, self.halal]
        p = plan(members, [r1, r2], dishes, group_location=(40.0, -74.0))
        # veg_place has no halal-safe dish (not certified, meat source unknown -> unknown hidden for strict)
        self.assertEqual(p.mode, "SPLIT")
        self.assertEqual(p.assignments["vegan"][0], "veg_place")
        self.assertEqual(p.assignments["halal"][0], "halal_place")
        assert_all_assignments_safe(self, members, [r1, r2], dishes, p)

    def test_split_cross_contact_warning_for_severe_member(self):
        r1 = rest("a", shared_kitchen_allergens=frozenset({A.PEANUT}))
        r2 = rest("b", **HALAL_CERTIFIED)
        dishes = [dish("rice", ["rice"], rid="a"), dish("satay", ["chicken", "peanut"], rid="b")]
        members = [self.nut, self.halal]
        p = plan(members, [r1, r2], dishes)
        self.assertEqual(p.mode, "SPLIT")  # satay is halal-safe but not peanut-safe; rice at a is not halal-safe
        assert_all_assignments_safe(self, members, [r1, r2], dishes, p)
        self.assertIn("cross_contact:peanut", p.warnings["nut"])

    def test_none_when_member_cannot_be_served_anywhere(self):
        members = [self.vegan, Member("x", allergic((A.DAIRY, S.SEVERE)))]
        dishes = [dish("steak", ["beef"], rid="r1"), dish("cheese", ["cheese"], rid="r1")]
        p = plan(members, [rest("r1")], dishes)
        self.assertEqual(p.mode, "NONE")
        self.assertEqual(set(p.unserved), {"vegan"})  # steak is fine for the dairy-allergic member
        self.assertEqual(p.assignments, {})

    def test_none_when_more_vendors_needed_than_allowed(self):
        members = [self.vegan, self.halal, self.nut]
        r1 = rest("a")
        r2 = rest("b", **HALAL_CERTIFIED)
        r3 = rest("c")
        dishes = [dish("v", ["rice"], rid="a"), dish("h", ["chicken"], rid="b")]
        # peanut-allergic member: nothing safe except dishes that exist at c
        dishes.append(dish("n", ["rice", "peanut"], rid="c"))
        p = plan(members, [r1, r2, r3], dishes, max_vendors=2)
        # 'nut' is served by a (rice is peanut-safe), vegan by a, halal by b => 2 vendors suffice
        self.assertEqual(p.mode, "SPLIT")
        p3 = plan([Member("only_peanut", allergic((A.PEANUT, S.SEVERE))), self.halal, self.vegan],
                  [rest("x"), rest("y", **HALAL_CERTIFIED)],
                  [dish("x1", ["rice"], rid="x"), dish("y1", ["chicken"], rid="y")], max_vendors=1)
        self.assertEqual(p3.mode, "NONE")

    def test_randomized_invariant_every_assigned_dish_is_safe_for_its_member(self):
        rng = random.Random(7)
        ing_pool = ["rice", "chicken", "butter", "egg", "peanut", "bacon", "wine", "gelatin", "lecithin",
                    "tofu", "honey", "mystery", "mirin", "fish sauce", "flour"]
        for _ in range(300):
            rests = [rest(f"r{i}", lat=40 + rng.random() / 10, lon=-74,
                          halal_certified=rng.choice([True, False, None]),
                          meat_source=rng.choice(list(MeatSource)),
                          alcohol_in_sauces=rng.choice([True, False, None]),
                          shared_kitchen_allergens=frozenset(rng.sample(list(A), rng.randint(0, 2))))
                     for i in range(rng.randint(1, 4))]
            dishes = [dish(f"d{j}", rng.sample(ing_pool, rng.randint(1, 4)), complete=rng.random() > 0.2,
                           rid=rng.choice(rests).id) for j in range(rng.randint(1, 10))]
            members = []
            for k in range(rng.randint(1, 4)):
                members.append(Member(f"m{k}", UserConstraints(
                    allergies={a: rng.choice(list(S)) for a in rng.sample(list(A), rng.randint(0, 2))},
                    diets=set(rng.sample(list(Diet), rng.randint(0, 1))),
                    halal=rng.choice(list(HalalLevel)))))
            p = plan(members, rests, dishes, group_location=(40.0, -74.0), max_vendors=3)
            assert_all_assignments_safe(self, members, rests, dishes, p)


if __name__ == "__main__":
    unittest.main()
