# Rollout-oracle implementation notes

## Repository safety and baseline

- Working remote: `https://github.com/GabaSatvik/learned-icbsh-mapf.git`
- Base/default branch: `main`
- Base commit: `ae4fa3571c5b5eb9d6378eebeb289831bb111543`
- Research branch: `research/deep-rollout-oracle`
- Reference remote inspected separately at commit `625cda82b341906af3df656e2da105970091bd66`.
- The reference checkout remained clean and was never configured as a working-repository remote.
- Base working tree was clean.
- Baseline `python -m mapf.test_features`: 3/3 legacy feature tests passed.
- Baseline unittest discovery: zero tests discovered.
- Baseline pytest: unavailable (`No module named pytest`). This is a pre-existing environment limitation.
- Baseline interpreter: Python 3.14.6. The full installed-package list is retained in the session log; principal numerical packages were NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, and XGBoost 3.4.1.

## Reference repository audit

The completed reference study uses a substantially revised solver architecture:

- `mapf.icbs_h.ICBShSolver.solve` owns the best-first high-level loop, heap OPEN ordering, CT expansion, pathmax, bypass, termination, telemetry, callbacks, and optional isolated-node continuation.
- `mapf.icbs_h.ICBSNode` stores per-agent constraints, paths, g/h/f, depth, pair counts, node ID, cached conflicts/classifications/conflict graph, flat constraints, and bypass count.
- `mapf.conflicts.detect_all_conflicts` and the `Conflict`/`Constraint` types implement collision and split representations.
- `mapf.cardinal.classify_conflicts` computes cardinal/semi/non-cardinal classifications and alternate equal-cost paths. `build_constraint_for_agent` creates each binary child constraint.
- `mapf.conflict_graph.CardinalConflictGraph.exact_mvc_size` and `compute_h4_and_graph` implement the exact cardinal-conflict-graph MVC heuristic.
- Child construction in `ICBShSolver.solve` recomputes child conflicts/MVC and applies `effective_h = max(child_h, parent.f - child_g)`, preserving pathmax.
- Strict eligible tiers are CARDINAL, then SEMI_CARDINAL, then NON_CARDINAL.
- `mapf.baselines.f_aware_residual_selector` implements the analytical residual selector; `_compute_candidate_lookahead_tuple` exposes its lexicographic key.
- `mapf.ml_revision_policy.candidate_pool` puts the residual winner first and chooses up to five other same-tier candidates by a frozen SHA-256 conflict hash.
- `mapf.ml_revision_policy.immediate_children` is the one-step teacher: ordinary two-way split, exact low-level replanning, full child classification/MVC, and pathmax. Its key is `(min effective child f, max effective child f)`, higher preferred.
- `mapf.features.FEATURE_NAMES` freezes the 24 Model B inputs. `FeatureExtractor.extract_all` produces them from parent/classification context only.
- `SavedLinear` loads the train-only-standardized Ridge artifact; `RevisionSelector` performs Model B scoring without online teacher leakage.
- `mapf.ml_revision.harvest`/the repair collectors save individual decision nodes, features, pools, and provenance. `final_study.py` runs isolated policy jobs and records capped timing/memory/solution telemetry.
- The final-study and ML-revision tests check pool determinism, child-teacher agreement, independent MVC values, timeout/infeasible distinctions, feature replay, no teacher leakage, and policy optimality on tiny instances.

The reference dataset contains 237 states and 1,064 complete candidate labels. Its stated split is families 08–19 for training, 20–23 for development, and 24–29 for the consumed final study. This experiment copied only the frozen train/development instance specifications (families 08–23); final-test inputs were not used.

## Working repository differences at the base commit

The original working `main` was an earlier, smaller implementation:

- `ICBS_H.solve` used a heap but had no pluggable selector, no strict same-tier candidate hook, no pathmax, no complete telemetry, no snapshot/continuation API, and only a simple logger callback.
- Conflict and constraint types/API names differed (`agent_1`/tuple constraints versus the final-study typed objects).
- A basic residual-like priority (first conflict in the best tier) existed only implicitly; the final analytical residual selector did not exist.
- Model B, its 24-feature schema, deterministic six-candidate pool, one-step verified teacher, and saved linear artifact did not exist.
- Existing features numbered 18 rather than 24.
- No saved teacher states, frozen family protocol, final-study runner, or full solver replay state existed.
- The remote `phase2-rollout-results` branch contains an older isolated child/subtree rollout. It skips MVC in descendants and converts capped runs to synthetic costs, so it was not reused as evidence for this study.

To establish exact comparability, the final-study versions of the solver core, residual selector, 24-feature extractor, one-step teacher, validation helpers, and Model B artifact were brought into this research branch. The legacy public aliases are retained where the final-study code provides them.

## Full-state finding

The old saved JSON state is not sufficient for global replay. It contains one CT node's paths, constraints, g/h, depth, bypass count, and pair counts, but not:

- the complete OPEN heap and its other live nodes;
- their cached conflicts/classifications/MVC graphs;
- the global node-ID counter and exact tie-ordering frontier;
- the live low-level path cache.

The reference solver itself documents that `initial_node` continuation does not restore the original global OPEN frontier. Therefore saved-node continuation is an `isolated_subtree` experiment and was not mixed with the primary results.

The new `on_decision_callback` pauses the live solver before selection and before state mutation. `capture_full_search_snapshot` deep-copies the current node, every other OPEN node, h-tables, path cache, node counter, and counters. Each candidate evaluation deep-copies that snapshot again, reinserts the current node into the heap, forces only its first eligible conflict, then uses the frozen residual selector. No mutable object is shared across candidates.

## Budget and censoring semantics

- One unit is one high-level CT pop/expansion.
- Generated CT nodes, low-level calls, OPEN size, MVC/lower bounds, bypasses, infeasible children, wall/CPU time, and plateau escape are recorded separately.
- A rollout stops before expansion `budget + 1`.
- A solved candidate records exact remaining expansions.
- An unsolved candidate is right-censored. Two jointly unsolved candidates are never tied and never form a fake strict pair.
- The per-candidate pilot safety cap was 15 wall seconds. Timeout is not infeasibility.
- `LB_0` is the minimum f across the captured current node and all other OPEN nodes. Plateau escape is the first completed expansion after which minimum OPEN f exceeds `LB_0`; missing escape is null/censored.

## State selection and reproducibility

- Smoke: first informative live residual-policy decision from five frozen train instances, one per encountered instance, budgets 50/100/250.
- Pilot: first informative live residual-policy decision from 40 distinct frozen train/development instances, budgets 50/100/250/500/1500.
- This avoids adjacent states from one solve, though it emphasizes early decisions.
- Candidate pool seed and Model B seed are 71/frozen by the reference implementation.
- Per-run metadata records Git branch/commit, dirty state, Python/platform, budgets, mode, continuation policy, and selection rule.

## Known limitations

The pilot has 25 fully complete state-level candidate sets; 15 remain partially censored. Short-horizon perfect ordering is based on directly observed early termination and cannot rank two jointly censored candidates. Timing was collected on one Windows host with candidate processes running in groups of up to four. Results cover only three known maps and synthetic frozen instances. These limitations motivate the decision gate in the final report.
