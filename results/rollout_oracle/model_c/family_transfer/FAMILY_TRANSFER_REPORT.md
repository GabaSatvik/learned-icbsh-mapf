# Model C family-transfer diagnosis

This analysis holds the 24 features and Ridge setup fixed. It uses only the existing training and development tables; the sealed test is untouched.

There are 19 nominal training families, but only 15 contain a strict conflict comparison. There are six development families, but only five are informative for pairwise accuracy. Most families contain only one to five states.

## Development families

| development family | Model C pairwise | Model B | residual | Model C regret |
|---|---:|---:|---:|---:|
| random-32-32-20/family-20 | 0.000 | 0.500 | 1.000 | 1.00 |
| random-32-32-20/family-21 | 0.467 | 0.433 | 0.533 | 2.80 |
| random-32-32-20/family-22 | 0.567 | 0.856 | 0.578 | 1.67 |
| room-32-32-4/family-22 | 0.389 | 0.622 | 0.856 | 1.00 |
| warehouse-32-32-aisles/family-21 | n/a | n/a | n/a | 0.00 |
| warehouse-32-32-aisles/family-22 | 0.500 | 1.000 | 1.000 | 2.00 |

## Marginal training-family influence

Positive delta means Model C improved when that training family was removed.

| family removed | mean accuracy delta | dev families improved | dev families worsened |
|---|---:|---:|---:|
| random-32-32-20/family-15 | +0.140 | 2 | 1 |
| random-32-32-20/family-10 | +0.140 | 2 | 2 |
| warehouse-32-32-aisles/family-13 | +0.100 | 1 | 0 |
| warehouse-32-32-aisles/family-16 | +0.096 | 1 | 1 |
| random-32-32-20/family-09 | +0.096 | 1 | 1 |
| room-32-32-4/family-19 | +0.091 | 1 | 2 |
| warehouse-32-32-aisles/family-17 | +0.062 | 1 | 0 |
| random-32-32-20/family-16 | +0.056 | 1 | 3 |
| room-32-32-4/family-10 | +0.029 | 1 | 1 |
| warehouse-32-32-aisles/family-08 | +0.016 | 1 | 1 |
| random-32-32-20/family-08 | +0.000 | 0 | 0 |
| random-32-32-20/family-18 | +0.000 | 0 | 0 |
| warehouse-32-32-aisles/family-18 | +0.000 | 0 | 0 |
| warehouse-32-32-aisles/family-19 | +0.000 | 0 | 0 |
| room-32-32-4/family-16 | -0.004 | 0 | 1 |
| room-32-32-4/family-13 | -0.004 | 0 | 1 |
| warehouse-32-32-aisles/family-11 | -0.004 | 0 | 1 |
| random-32-32-20/family-12 | -0.007 | 0 | 1 |
| random-32-32-20/family-11 | -0.009 | 0 | 2 |

## Feature/label consistency

Nearest-pair label agreement across training families: 0.498.
Development pair agreement with its nearest training pair: 0.378.

Median nearest-pair distance is 3.642 when the label agrees and 3.368 when it disagrees. Distance therefore does not identify trustworthy transfers in this sample.

Low agreement means conflict pairs that look similar in the frozen feature space can still have opposite deep-rollout preferences. Nearest-neighbor distance is only diagnostic in this 24-dimensional standardized space.

Most family-dependent feature directions: conflict_time, goal_dist_i, goal_dist_j, downstream_conflict_count_j, downstream_conflict_count_i, agent_j_path_length, marginal_conflict_delta_min, agent_i_path_length.

## Interpretation

Model C beats residual on 0 of 5 informative development families, so the failure is not confined to one held-out family. Removing some training families helps some development families but hurts others; no universally harmful family was found. A notable interaction is warehouse family 13: it transfers best when trained alone, yet the full model improves when it is removed. That is evidence of unstable mixing, not evidence that its data should be discarded.

The chance-level cross-family consistency and feature-direction reversals support a frozen-feature/context mismatch: deep rollout quality depends on search context that the 24 local features do not encode consistently. The very small number of independent states per family limits certainty, so the correct response is a targeted feature-and-coverage experiment, not blind data collection and not deletion of inconvenient families.

No production selector was changed and no new model was accepted.