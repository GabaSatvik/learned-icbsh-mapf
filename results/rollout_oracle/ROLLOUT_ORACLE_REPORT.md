# Deep rollout oracle pilot

## Result

The pilot captured 40 live full-OPEN decision states and 176 candidates. All candidates completed in 25 states; 19 of those states had nonzero downstream expansion gaps. All completed candidates preserved the same optimal solution cost within their state.

The one-step teacher was worse than residual on this sample. Deep horizons identified the eventual best candidate whenever at least one candidate terminated within the horizon, but this is direct partial observation of time-to-solution; it does not rank two jointly censored candidates.

## Ranking and regret

| policy | comparable pairs | pairwise | family-macro pairwise | top-1 | mean regret | worst regret | covered states |
|---|---:|---:|---:|---:|---:|---:|---:|
| residual | 108 | 0.796 | 0.718 | 0.720 | 0.720 | 7.000 | 25/25 |
| one_step | 65 | 0.677 | 0.703 | 0.680 | 1.040 | 8.000 | 25/25 |
| rollout_50 | 72 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 18/25 |
| rollout_100 | 82 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 20/25 |
| rollout_250 | 90 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 22/25 |
| rollout_500 | 93 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 24/25 |
| rollout_1500 | 108 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 25/25 |
| plateau_escape | 39 | 0.795 | 0.694 | 0.762 | 0.762 | 7.000 | 21/25 |

## Horizon quality and cost

| budget | candidate completion | exact-state selector coverage | mean wall s/candidate |
|---:|---:|---:|---:|
| 50 | 0.352 | 0.720 | 2.111 |
| 100 | 0.403 | 0.800 | 4.028 |
| 250 | 0.443 | 0.880 | 7.027 |
| 500 | 0.472 | 0.960 | 8.309 |
| 1500 | 0.506 | 1.000 | 8.400 |

The 50-expansion horizon is the best observed signal/cost point for early completions, but coverage rises from 72.0% at 50 to 100.0% at 1500. The 15-second cap flattens measured cost at long horizons.

## Plateau escape

Across 78 exact candidates with observed escape, Spearman(escape expansions, eventual expansions) = -0.245. The plateau selector results are in the ranking table. Censored escape times were not assigned numeric values.

## Representative one-step counterexamples

- room-32-32-4/family-10/k16/deep-q0: teacher chose 0-3-vertex-14-20-9 (23 expansions) over 2-3-vertex-1-27-15 (15), regret 8; keys [385.0, 386.0] vs [385.0, 385.0].
- random-32-32-20/family-15/k24/deep-q0: teacher chose 12-14-vertex-30-3-20 (87 expansions) over 13-14-edge-10-21-20-22-20 (80), regret 7; keys [464.0, 488.0] vs [464.0, 464.0].
- warehouse-32-32-aisles/family-11/k32/deep-q0: teacher chose 22-28-vertex-9-8-14 (161 expansions) over 13-19-vertex-6-16-5 (156), regret 5; keys [627.0, 632.0] vs [625.0, 626.0].
- random-32-32-20/family-21/k24/deep-q0: teacher chose 1-16-vertex-24-4-7 (464 expansions) over 3-14-vertex-14-16-7 (462), regret 2; keys [510.0, 510.0] vs [510.0, 510.0].
- random-32-32-20/family-16/k32/deep-q0: teacher chose 9-26-vertex-1-18-24 (447 expansions) over 3-22-vertex-7-27-20 (446), regret 1; keys [659.0, 659.0] vs [658.0, 659.0].

## Decision gate

- A. Conflict choices materially affect effort: **yes**, in 19/25 complete states.
- B. Multi-step rollout gives a better observable signal: **yes for early-completing candidates**, but jointly censored ranking remains unresolved.
- C. Best observed tradeoff: **50 expansions for cheap early-completion signal; 500–1500 for coverage**.
- D. Plateau escape: **not established as useful** (candidate-level Spearman -0.245).
- E. Ready for a new ranker: **not yet**; more exact/censoring-aware oracle data is needed before changing labels.

Final recommendation: **PROMISING BUT NEED MORE ORACLE DATA**.

## Limitations

- 15/40 states lack full candidate completion under the 1500-expansion/15-second cap.
- Short-horizon perfect accuracy is a censoring-aware early-termination result, not a learned prediction among jointly unsolved candidates.
- One early decision per family reduces adjacency correlation but emphasizes root/near-root behavior.
- Only the three known maps and synthetic frozen train/development instances are covered.