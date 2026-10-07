# Project 6 — Full Execution Plan
### Reading → skills gap → implementation → data → checkpoints → the bar for calling this a paper

---

## 1. Reading Plan — what, and how long

You've read the three assigned papers already, so this is about closing specific gaps, not starting over. Split by priority so the team doesn't all read the same things.

| Reading | Why | Time | Priority |
|---|---|---|---|
| **Boyarski et al., 2015 — "ICBS: Improved Conflict-Based Search"** | The cardinal/semi-cardinal/non-cardinal classification and bypassing — the mechanism our whole project sits on top of — is only described *secondhand*, in Section 2.1 of the ICBS-h paper. Read the original. | ~1 hr | **Must** — one person, before Week 1 ends |
| **Stern et al., 2019 — "Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks"** | This is the actual benchmark-suite paper every one of our three papers draws maps from. Defines the standard file formats, evaluation protocol, and terminology you'll be expected to use correctly. | ~45 min | **Must** — whoever owns data/benchmarking |
| **XGBoost's Learning-to-Rank docs** (`rank:pairwise`, `LambdaMART`) | You need to know what the objective is actually optimizing, not just the API call — see the "pairwise ranking" defense question in the Q&A guide. | ~1 hr hands-on | **Must** — whoever owns the model |
| **Li et al., 2021 — "Anytime MAPF via Large Neighborhood Search"** | The actual MAPF-LNS paper ADDRESS builds on — gives more depth on the destroy-heuristic mechanics ADDRESS only summarizes. Not load-bearing for our method (we're CBS-based, not LNS-based), but valuable if anyone gets asked to go deeper on ADDRESS. | ~1.5 hr | Should — optional but recommended for whoever presents Part 2 of the script |
| **Chapelle & Li, 2011 — "An Empirical Evaluation of Thompson Sampling"** | Only needed if you pursue the week-8 stretch idea (bandit correction layered on the ranker). | ~45 min | Optional |

**Total team time: ~5 hours**, split across three people rather than everyone reading everything. Close it out with a **30-minute sync** where each person explains what they read to the other two — that's also a free extra rehearsal of the "explain it out loud" skill the Q&A guides are training.

---

## 2. What We Need to Add From Our Side — the honest skills gap

Answer these three questions as a team before Week 1 is over, because the answers change the implementation plan in Section 3:

1. **Has anyone actually implemented a graph search algorithm (A\*, Dijkstra) from scratch before, in any language?** If no one has, budget an extra 2–3 days in Week 1 specifically for the low-level single-agent search before attempting the high-level CT — don't try to build both at once.
2. **Has anyone used XGBoost/LightGBM's ranking objectives before**, as opposed to standard classification/regression? If not, the ~1 hr doc read above plus a 30-minute toy example (rank three items by two made-up features) before touching real data will save real debugging time later.
3. **Does anyone have experience with priority queues / heaps in your chosen language?** CBS's OPEN list is a min-heap by cost — this is a small thing, but "reinventing sorted-list behavior with a plain list" is a classic way to quietly make your CT implementation $O(n^2)$ per pop instead of $O(\log n)$, which will bite you exactly when you scale up to the larger maps in Week 7.

None of these are blockers — they're just things to name explicitly now, as a 15-minute team conversation, rather than discovering them mid-sprint.

---

## 3. Implementation & Starting Plan

### The core decision: build our own CBS/ICBS-h, in Python, from the papers' own pseudocode

This is a deliberate recommendation, not a default. Two real implementation paths exist:

- **Adapt an existing open-source implementation.** A strong one exists: [**Jiaoyang-Li/CBSH2-RTC**](https://github.com/Jiaoyang-Li/CBSH2-RTC) on GitHub — C++, implements CBS with prioritizing conflicts, bypassing, and high-level heuristics (an even more advanced WDG heuristic than ICBS-h's vertex cover). Worth knowing: **Jiaoyang Li is a co-author on our actual assigned ICBS-h paper.** This is about as authoritative a reference implementation as exists for this exact algorithm.
- **Build our own, in Python, following Algorithm 1 in Sharon et al. (2012) and the mechanisms in Felner et al. (2018) directly.**

**Recommendation: build our own.** Reasoning, not just preference:
- Our entire contribution is a new hook *inside* the conflict-selection step — we need to add feature-logging at exactly that point. Retrofitting that into an unfamiliar C++ codebase (Boost dependencies, existing WDG/symmetry-reasoning machinery we don't need) costs more Week 1–2 time than writing a clean, purpose-built version ourselves.
- The pseudocode is genuinely complete enough to implement directly — Sharon et al.'s Algorithm 1 is the whole high-level loop, and ICBS's cardinal/semi/non-cardinal test plus ICBS-h4's vertex-cover computation are both precisely specified in the papers you already have.
- It directly demonstrates the thing your professor is testing for: implementing the paper, not importing someone else's implementation of it.
- **Use CBSH2-RTC as a validation oracle, not a base** — see Checkpoint 1 below. Clone it, run it on the same benchmark instances, and use its output costs/node counts as ground truth to sanity-check that *our* implementation is actually correct.

### Rough module structure (Python)

```
mapf/
  grid.py          # map parsing (movingai .map format), 4-connected graph
  low_level.py      # space-time A* for one agent given a constraint set
  high_level_cbs.py # plain CBS: OPEN as heapq, split on first conflict found
  cardinal.py        # ICBS: classify conflict as cardinal/semi/non-cardinal, bypass logic
  conflict_graph.py  # build G_CF, min vertex cover (exact, small graphs — brute force/ILP is fine at this scale)
  icbs_h.py           # ICBS-h4: g+h priority ordering
  features.py         # per-conflict feature extraction (the 6 features)
  logger.py           # instrumentation: log every split decision + features + outcome
  rank_model.py        # training script: XGBoost rank:pairwise over logged data
  learned_cbs.py        # final integration: ICBS-h + learned tie-break among cardinal ties
  eval.py               # benchmark harness: run N scenarios, report nodes/runtime/success rate
```

### Week-by-week

- **Week 1:** Reading sync (Section 1) → map parser + low-level space-time A* → plain CBS high level, correctness-tested on small hand-built instances (2–3 agents you can solve on paper).
- **Week 2:** Cardinal/semi/non-cardinal classification + bypassing (ICBS) → conflict graph + exact min vertex cover → ICBS-h4 node-ordering. Validate against CBSH2-RTC on the same small/medium maps.
- **Week 3:** Feature extraction module + logging instrumentation. Hand-weighted linear scorer as the cheap first test (see Checkpoint 3).
- **Week 4:** Run the linear-scorer experiment at real scale; decide go/no-go on the full ML pipeline based on the result, not the calendar.
- **Week 5–6:** Label generation (proxy or short-rollout, per your earlier decision), train XGBoost ranker + logistic baseline + random-ranker control, integrate into `learned_cbs.py`.
- **Week 7:** Full benchmark sweep across maps and agent counts, mirroring the source papers' own table structure.
- **Week 8:** Ablation, ablation write-up, report.

---

## 4. Where We'll Actually Get Data

Two genuinely different things are called "data" here — keep them separate in your own heads and in the report, or you'll get a confused question about it.

### (a) MAPF problem instances — external, from the exact source these papers use

Every map used across all three of your papers — and every map in ADDRESS's experiments specifically — comes from **movingai.com's MAPF benchmark repository**: `https://movingai.com/benchmarks/mapf/index.html`. I checked it directly, and the vertex counts confirm it's the exact same data: `den520d` = 28,178 states, `ost003d` = 13,214, `warehouse-20-40-10-2-2` = 38,756, `Paris_1_256` = 47,240, `random-32-32-20` = 819 — these match the numbers reported in the ADDRESS paper's own Figure 6 exactly. That's the site.

- Each map ships as a `.map` file plus 25 "random" and 25 "even" `.scen` scenario files — a scenario file is a list of start/goal pairs; you take the first $k$ rows to get a $k$-agent instance. Format is documented at `movingai.com/benchmarks/formats.html`.
- For the CBS paper's own small 8×8 obstacle-density experiments, you don't need external data at all — those are synthetic, randomly-obstacled grids you generate yourselves (the paper literally specifies "10% to 35% randomly-placed obstacles" — a ~20-line generator).

### (b) ML training data — internal, self-generated, not downloadable from anywhere

This is the part that doesn't exist yet and can't be fetched — it's produced by running *your own instrumented ICBS-h implementation* on (a)'s instances and logging every branch decision, per Section 3's `logger.py`. This is the actual Week 3–6 work, not a data-collection errand.

---

## 5. How to Check Progress — concrete, falsifiable checkpoints

Vague "week 3: build features" milestones don't actually tell you if you're on track. Use pass/fail bars you can check against the papers' own published numbers wherever possible:

- **Checkpoint 1 (end Week 1):** Plain CBS returns *provably optimal* solutions on 5+ small hand-solvable instances (work them out by hand first). Then run the same small/medium benchmark instances through cloned CBSH2-RTC and compare solution costs — they must match exactly (both are optimal solvers; a mismatch means a bug in yours, not a modeling choice). This is a hard pass/fail bar, not a vibe check.
- **Checkpoint 2 (end Week 2):** ICBS-h4 vs. plain CBS on an 8×8-style grid with a few obstacle densities shows the *same directional trend* the ICBS-h paper reports (fewer nodes expanded, more pronounced at higher obstacle density). You're not trying to match their exact numbers — different codebase, different constant factors — but if your trend runs *backwards*, something's wrong.
- **Checkpoint 3 (Week 3–4, the real go/no-go gate):** the hand-weighted linear scorer beats ICBS-h's arbitrary cardinal-tie-break by some pre-agreed minimum margin (pick this number as a team *now*, e.g. "≥10% median node reduction across ≥30 instances," so you can't retroactively lower the bar if the result disappoints). If it fails this bar, that's a legitimate, reportable outcome — write up *why*, don't quietly keep pushing forward as if nothing happened.
- **Checkpoint 4 (Week 5–6):** trained ranker beats *three* things simultaneously — plain ICBS-h, the linear baseline, and a randomly-initialized ranker (Section 3.3 of the Deep-Verification guide). Beating only the weakest of the three isn't a result.
- **Checkpoint 5 (Week 7):** a complete results table in the same format as the source papers' own tables (mean ± 95% CI, multiple scenarios per map, multiple agent counts) — so a reader familiar with those papers can put your table next to theirs and understand it immediately.
- **Checkpoint 6 (Week 8):** every claim in the report traces to a specific figure/table you can point to — no claim that only lives in someone's memory of "yeah it worked better."

---

## 6. What to Strive For — making this genuinely paper-shaped

You have an unusually good template sitting right in front of you: your three source papers *are* the exact structure and rigor bar to match. Don't invent a report format — mirror theirs.

**Structure:** Abstract → Introduction (the problem, framed exactly like Part 1 of your script) → Related Work (Part 2) → Method (Part 3, but with the full ML detail) → Experimental Setup → Results → Discussion/Limitations → Conclusion. This is literally the section list of all three papers you've been studying.

**Rigor checklist — copy these habits directly from your sources, don't just aim for "good enough":**
- **Multiple scenarios per map, not one.** ADDRESS uses 25 random scenarios per map specifically so a single lucky/unlucky instance can't dominate the result — do the same.
- **Report confidence intervals, not single numbers** — every results table in ADDRESS is mean ± 95% CI. A bare average invites the question "how much does this vary?"
- **Disclose hardware and time budget explicitly** — every paper you've read does this (down to the CPU model and RAM). It's what makes a result reproducible.
- **Full baseline set, not a strawman** — plain CBS, ICBS-h4, our linear scorer, our random-ranker control, our trained ranker. Comparing only against the weakest baseline is the fastest way to lose credibility with someone who's read the source papers closely.
- **An honest limitations section** — you already have the material for this from the "will it really improve" discussion and the E1/E4 questions in the Deep-Verification guide. State plainly where you'd predict it *won't* help (sparse maps, single-cardinal-conflict-per-node scenarios) rather than letting a reader find the gap themselves.
- **Ablation on the 6 features** — you already planned this; make sure it actually ships, not just gets mentioned as future work.

**How far this could realistically go:** if the Week 3–4 checkpoint passes and the full pipeline holds up under the rigor checklist above, this is genuinely shaped like a short paper for a venue like **SoCS (Symposium on Combinatorial Search)** — the natural community home, since it's where several works cited across your own three papers were published — or at minimum a clean arXiv writeup. Say this to yourselves as a real possibility, not a guarantee: the difference between "solid course project" and "something we could actually submit somewhere" is entirely the rigor checklist above, not the idea itself. The idea is already good enough; execution and honesty about what did and didn't work is what would make it real.
