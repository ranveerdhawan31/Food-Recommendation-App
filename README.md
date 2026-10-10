# foodrec: personalized food recommendation (safety-first)

Flow: survey -> safety filter -> taste filter -> retrieval -> ranking -> group fusion -> order routing -> feedback/MLOps

Covers mains, desserts and non-alcoholic beverages (`pipeline.recommend_meal`).

## Run
    python -m unittest discover -s tests -t .      # stdlib + numpy (+ cryptography for one privacy test)

## Layout (build order)
| Module | Role |
|---|---|
| `schema.py` | Data model. Hard constraints live only in `UserConstraints`. |
| `knowledge_graph.py` | Hidden-ingredient graph (casein->dairy, Worcestershire->fish, gelatin->pork/beef, lecithin->soy/egg). |
| `inference.py` | Rule-based ingredient extraction with confidence (swap in an NER/LLM model). |
| `safety.py` | Deterministic SAFE/UNKNOWN/UNSAFE labeling: allergens, diets, Halal, beverage alcohol policy. `verification_warning()` builds the "Verify with the restaurant" text. |
| `group_safety.py` | Union constraints, shared pool, single-vendor and split-order planning. |
| `survey.py` | Hard constraints asked explicitly; soft tastes chosen by information gain. `NA_CUISINES`, `SoftSurvey.declare` (stated likes/avoids), `build_taste_preferences` (sweet, dessert, beverage, caffeine). |
| `taste.py` | Category, avoided-cuisine and caffeine filters. Run after safety; can only remove dishes. |
| `retrieval.py` | Geo/hours/budget/ETA filters + embedding search (`VectorIndex` seam for pgvector). |
| `ranking.py` | Taste/context features only (now incl. `sweetness_fit`, `tag_affinity`), heuristic ranker, bandit noise, MMR. |
| `pipeline.py` | Single-user flow. Safety runs before retrieval and ranking. `recommend(category=...)` and `recommend_meal()`. Each `Recommendation` carries `.warning`. |
| `group_fusion.py` | Least-misery / Nash aggregation, fairness ledger, shuffle/vote session. |
| `routing.py` | `MenuProvider` interface, quote ranking, deep links, order-time safety re-check. |
| `mlops.py` | Feedback events, allergy-incident quarantine, safety suite, deployment gate. |
| `privacy.py` | Approximate location, consent gate, field encryption. |
| `api.py` | FastAPI WebSocket wrapper (not run in the build sandbox). |

## Safety invariants (all tested)
- No ranking feature mentions allergies, diet or Halal (`assert_no_safety_features`).
- An adversarial ranker cannot resurface a blocked dish, even with large exploration noise.
- Every dish shown or assigned to a member passes that member's own constraints (300 randomized group scenarios).
- Group sessions reject dish ids outside the safe pool.
- Every allergen-bearing ingredient in the graph is UNSAFE for a user with that allergen.
- Deployment gate: any safety violation blocks release regardless of conversion or convergence gains.

- Beverages are only ever shown when the provider sets `Dish.alcohol_free=True` AND no ingredient resolves to alcohol. A missing flag hides the drink for every user (fail closed).
- Avoided cuisines, caffeine-free and category choices can never add a dish the safety layer removed (randomized 200-scenario test over mains, desserts and drinks).
- Any dish with an unidentified ingredient or no ingredient data carries a "Verify with the restaurant" warning, including for users with no hard constraints.

## Decisions you may want to change
- Verification for users with NO hard constraints fires only for unidentified ingredients or missing ingredient data, not for merely incomplete-but-identified lists (constrained users keep the broader rule).
- `vanilla extract` and `kombucha` are tagged as alcohol (small but real amounts). Strict Halal users never see dishes listing them, and they are excluded from beverages for everyone. Say "vanilla bean" or "vanilla" in provider data to avoid this.
- Chocolate is `possible` dairy + soy (UNKNOWN for those allergies). Marshmallow contains gelatin, so it is non-vegetarian and possible-pork.
- Caffeine-free is a taste filter: only dishes KNOWN to contain caffeine are dropped. Do not use it for a medical need.
- `recommend()` now defaults to `category="main"`; pass `category=None` for the old all-dishes behaviour.
- New ranking weights (`sweetness_fit` 1.2, `tag_affinity` 1.0) are untuned guesses.
- Strict Halal users also have UNKNOWN dishes hidden (spec only required this for severe allergies): `UserConstraints.exclude_unknown`.
- Declared shared-kitchen allergens produce a warning, not an exclusion, for severe allergies.
- Detection below 0.5 confidence is UNKNOWN for mild/moderate allergies but UNSAFE for severe ones.
- Verify label threshold is 0.9 (`safety.VERIFY_THRESHOLD`).

## Not done / needs your input
- `FEATURE_NAMES` grew by two features: any `XGBRanker` booster trained on the old 8 features must be retrained.
- Group planning (`group_safety`, `group_fusion`) is not category-aware yet, and `verification_warning` is not wired into group plans.
- `NA_CUISINES` is a judgment list with no popularity priors (invented numbers would be worse than none). Fit priors from real data.
- The allergen enum has 9 entries. Canada's priority list also includes mustard and sulphites, which are not modelled.
- The knowledge graph is a ~150-entry seed; curate it against a real allergen database before launch.
- Two-tower embeddings and the ranker are untrained baselines (feature hashing, linear weights). `XGBRanker` is an untested adapter.
- Delivery partner access, real menu APIs and deep-link formats come from your partner agreements; `routing.py` takes URL templates as config.
- PostgreSQL/pgvector, Redis session state, SageMaker training and AWS deployment are not wired.
- This is a software filter, not a medical guarantee. Keep the "verify with restaurant" prompts for allergy results.
