# Model C — fixed Ridge with rollout labels

Training uses 46 queries, 168 candidates, and 188 strict pairs. Validation never enters the fit.

| split | policy | family-macro pairwise | top-1 | mean regret | worst regret |
|---|---|---:|---:|---:|---:|
| train | model_c | 0.706 | 0.717 | 2.543 | 29 |
| train | model_b | 0.543 | 0.609 | 2.783 | 23 |
| train | residual | 0.623 | 0.630 | 3.609 | 60 |
| train | one_step | 0.597 | 0.652 | 2.304 | 26 |
| validation | model_c | 0.384 | 0.429 | 1.786 | 10 |
| validation | model_b | 0.682 | 0.571 | 0.857 | 3 |
| validation | residual | 0.793 | 0.714 | 0.643 | 3 |
| validation | one_step | 0.636 | 0.643 | 1.714 | 14 |

## Development gate

- zero_serialization_error: **True**
- validation_family_macro_beats_model_b: **False**
- validation_family_macro_beats_residual: **False**
- validation_mean_regret_beats_model_b: **False**

Decision: **STOP_MODEL_C_FAILED_DEVELOPMENT_GATE**.

The fresh sealed test manifest was generated before fitting and was not solved or inspected.