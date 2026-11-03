# Smoke rollout-oracle report

Five live train-split decisions were evaluated across random, room, and warehouse maps with 2–6 candidates per state and budgets 50/100/250.

1. **Determinism:** yes. The dedicated determinism, prefix, residual-equivalence, and isolation tests pass; repeated non-timing outputs match.
2. **Candidate variability:** yes, but limited in the smoke. Three states completed every candidate by 250 expansions and one had a nonzero expansion gap.
3. **Difference from one-step teacher:** the small fully observed subset did not establish a difference in aggregate accuracy; the pilot was needed.
4. **Difference from residual:** likewise not established in the five-state subset.
5. **Full completion:** yes. All candidates completed in 3/5 states; hard room-map candidates remained capped.
6. **Optimal cost:** preserved for every completed candidate, with one common cost per state.
7. **Practicality:** feasible. Easy candidates finish quickly; hard room candidates can consume the wall cap, motivating independent candidate processes and explicit timeout censoring.
8. **Correctness bugs:** none remained after smoke. A copied Model B artifact initially failed its checksum because text normalization changed the file bytes; the working metadata now hashes the exact committed artifact. No rollout or solver correctness failure occurred.

The smoke gate passed, so the 40-state train/development pilot proceeded. Capped candidates were retained as right-censored observations and were not assigned budget-valued labels.
