# Adaptive oracle-data campaign

## Stage A — recover the censored pilot states

Target the 15 pilot states in which no candidate completed under the original
1,500-expansion / 15-second cap (87 candidates total).

- Recapture the same first informative live decision deterministically.
- Require the candidate conflict IDs to match the original pilot exactly.
- Preserve full-search-state mode and residual continuation.
- Run horizons 1,500, 3,000, and 5,000 CT expansions.
- Stop a candidate immediately when solved.
- Apply a 60-second wall cap per candidate; timeout remains right-censoring.
- Run at most four independent candidates concurrently.
- Write one append-safe part file per state, then merge.

Stage A succeeds if it materially increases fully observed state sets without an
optimal-cost mismatch. It does not retroactively replace the original raw pilot.

## Stage B — add distinct states

Only after Stage A is analyzed:

- harvest additional train/development states, never final-test families;
- prefer previously unused instances first;
- then, if required, use a later decision at least five informative decisions
  away from the first captured decision;
- stratify across maps and agent counts;
- use 50/100/250/500 initially and extend only unresolved candidates;
- keep exact completion, known solved-vs-censored comparisons, and progress
  proxies as separate label classes.

## Gate before ML

Do not train until all of the following are met:

- at least 60 fully observed state-level candidate sets;
- at least 70% candidate completion or an independently validated
  censoring-aware comparison method;
- no optimal-cost disagreement;
- family-macro rollout preference improves on residual;
- evidence is not limited to candidates that finish within the shortest horizon.
