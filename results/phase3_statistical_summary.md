# Phase 3 Statistical Benchmark Summary (25 Scenarios/Cell, 6 Solvers)

| Map | k | Solver | Solved | Median Nodes | Mean ± 95% CI Nodes | Median Runtime (s) | Wilcoxon p vs Learned_CL |
|---|---|---|---|---|---|---|---|
| `random-32-32-20` | 20 | `PlainCBS` | 22/25 | 28.0 | 77.8 ± 43.3 | 0.024s | 0.0000 * |
| `random-32-32-20` | 20 | `PlainICBSh` | 23/25 | 7.0 | 23.2 ± 17.4 | 0.013s | 0.8300 |
| `random-32-32-20` | 20 | `RandomRankerICBSh` | 23/25 | 8.0 | 24.6 ± 18.3 | 0.015s | 0.5834 |
| `random-32-32-20` | 20 | `DisjointSplittingICBSh` | 23/25 | 7.0 | 21.6 ± 16.3 | 0.012s | 0.9456 |
| `random-32-32-20` | 20 | `FAwareResidualICBSh` | 22/25 | 7.0 | 24.4 ± 18.6 | 0.013s | 0.9101 |
| `random-32-32-20` | 20 | `Learned_CL_ICBSh` | 22/25 | 7.0 | 24.7 ± 18.6 | 0.044s | — |
| `random-32-32-20` | 28 | `PlainCBS` | 7/25 | 350.0 | 283.5 ± 48.3 | 0.279s | 0.0000 * |
| `random-32-32-20` | 28 | `PlainICBSh` | 20/25 | 28.0 | 57.0 ± 21.0 | 0.135s | 0.4660 |
| `random-32-32-20` | 28 | `RandomRankerICBSh` | 20/25 | 28.0 | 58.3 ± 21.5 | 0.108s | 0.0622 |
| `random-32-32-20` | 28 | `DisjointSplittingICBSh` | 20/25 | 20.0 | 54.0 ± 21.5 | 0.098s | 0.9308 |
| `random-32-32-20` | 28 | `FAwareResidualICBSh` | 20/25 | 20.0 | 56.5 ± 22.1 | 0.090s | 0.9308 |
| `random-32-32-20` | 28 | `Learned_CL_ICBSh` | 19/25 | 27.0 | 58.2 ± 22.5 | 0.185s | — |
| `room-32-32-4` | 16 | `PlainCBS` | 19/25 | 65.0 | 128.0 ± 56.7 | 0.054s | 0.0000 * |
| `room-32-32-4` | 16 | `PlainICBSh` | 20/25 | 12.0 | 40.1 ± 22.9 | 0.031s | 0.0141 * |
| `room-32-32-4` | 16 | `RandomRankerICBSh` | 20/25 | 14.0 | 39.6 ± 22.5 | 0.032s | 0.0058 * |
| `room-32-32-4` | 16 | `DisjointSplittingICBSh` | 20/25 | 8.0 | 37.0 ± 22.8 | 0.022s | 0.1425 |
| `room-32-32-4` | 16 | `FAwareResidualICBSh` | 20/25 | 8.0 | 36.9 ± 22.8 | 0.026s | 0.3274 |
| `room-32-32-4` | 16 | `Learned_CL_ICBSh` | 20/25 | 9.0 | 36.8 ± 22.8 | 0.070s | — |
| `room-32-32-4` | 20 | `PlainCBS` | 13/25 | 192.0 | 200.7 ± 60.0 | 0.168s | 0.0000 * |
| `room-32-32-4` | 20 | `PlainICBSh` | 18/25 | 30.0 | 63.0 ± 24.2 | 0.106s | 0.0130 * |
| `room-32-32-4` | 20 | `RandomRankerICBSh` | 18/25 | 31.0 | 61.2 ± 23.9 | 0.113s | 0.0034 * |
| `room-32-32-4` | 20 | `DisjointSplittingICBSh` | 19/25 | 18.0 | 54.4 ± 23.8 | 0.067s | 0.0328 * |
| `room-32-32-4` | 20 | `FAwareResidualICBSh` | 19/25 | 18.0 | 54.1 ± 23.8 | 0.065s | 0.0721 |
| `room-32-32-4` | 20 | `Learned_CL_ICBSh` | 19/25 | 18.0 | 53.7 ± 23.7 | 0.119s | — |
