# Model C family-level learning curve

Fixed Ridge, fixed 24 features, alpha 10, 30 deterministic map-stratified nested family orders. The six development families are never used for fitting.

| training families | mean train queries | mean strict pairs | dev family-macro accuracy | std | dev top-1 | dev mean regret |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 9.1 | 39.6 | 0.500 | 0.092 | 0.505 | 9.379 |
| 8 | 18.0 | 78.9 | 0.489 | 0.105 | 0.460 | 6.731 |
| 12 | 27.6 | 121.7 | 0.499 | 0.086 | 0.490 | 5.481 |
| 16 | 37.6 | 156.5 | 0.466 | 0.068 | 0.462 | 1.881 |
| 19 | 46.0 | 188.0 | 0.384 | 0.000 | 0.429 | 1.786 |

Linear trend: -0.0064 accuracy per added family.
Change from 4 to all 19 families: -0.115.
Best mean development accuracy occurs at 4 families.

Diagnosis: **NO_POSITIVE_LEARNING_CURVE_EVIDENCE**.

Interpretation: more families reduce mean development regret, but they do not improve the primary ranking-accuracy measure. The all-family model is worse than the average smaller-family model on the fixed development set. This is not evidence that four families are sufficient or preferable; the development set has only six families, and the smaller-subset estimates vary substantially with which families are sampled.

The result does not support the simple explanation that Model C failed only because it had too few examples. It points instead toward family/distribution mismatch and/or a limitation of the fixed 24-feature linear representation. More data could still help if it is deliberately diverse and targeted, but blindly collecting more states from the same family mixture is not justified.

This diagnoses data quantity only for the current feature representation and linear model. It does not open or evaluate the sealed test set.