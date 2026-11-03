# Adaptive oracle-data campaign report

The combined campaign contains 75 states and 302 candidates. 60 state sets (215 candidates) are fully observed. Overall candidate completion is 71.2%.

49/60 exact states show a nonzero downstream expansion gap. No optimal-cost mismatch was observed.

## Policy comparison on exact states

| policy | pairs | pairwise | family-macro | top-1 | mean regret | family-macro regret | worst |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual | 250 | 0.648 | 0.660 | 0.650 | 2.917 | 1.568 | 60 |
| one_step | 171 | 0.579 | 0.637 | 0.650 | 2.167 | 1.544 | 26 |
| model_b | 250 | 0.532 | 0.533 | 0.600 | 2.333 | 1.768 | 23 |
| rollout_50 | 119 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |
| rollout_100 | 159 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |
| rollout_250 | 188 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |
| rollout_500 | 228 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |
| rollout_1500 | 250 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0 |

The perfect rollout rows are censoring-safe direct observations: a horizon only covers a state when at least one candidate has solved. Coverage grows from 33/60 states at 50 expansions to 60/60 at 1,500; jointly unsolved candidates are still not ranked.

## Data-gate result

- At least 60 exact states: **True**
- At least 70% candidate completion: **True**
- Optimal-cost preservation: **True**
- Hard-state recovery: 0/3 states completed after the 60-second extension.
- Plateau correlation: 0.162; still not supported as a standalone objective.

The three hard-state recovery attempts produced no exact labels and are retained as censored evidence. The 35 new exact states are spaced at q5/q10/q15/q20, but several come from the same family trajectories; family-macro results must therefore remain primary.

## Recommendation

**PROCEED WITH ROLLOUT-BASED ML TRAINING**, while holding the model class and 24-feature schema fixed. Train only on exact strict pairs and known solved-vs-censored comparisons; do not create labels between jointly censored candidates. The consumed final test families must remain excluded.