# Phase 3 Statistical Benchmark Summary (25 Scenarios/Cell, 6 Solvers)

| Map | k | Solver | Solved | Median Nodes | Mean ± 95% CI Nodes | Median Runtime (s) | Wilcoxon p vs Learned_CL |
|---|---|---|---|---|---|---|---|
| `random-32-32-20` | 20 | `PlainCBS` | 21/25 | 1317.0 | 4089.7 ± 2548.7 | 5.196s | 0.0000 * |
| `random-32-32-20` | 20 | `PlainICBSh` | 25/25 | 37.0 | 72.2 ± 45.4 | 2.497s | 0.0002 * |
| `random-32-32-20` | 20 | `RandomRankerICBSh` | 18/25 | 96.0 | 230.1 ± 94.5 | 5.255s | 0.0000 * |
| `random-32-32-20` | 20 | `DisjointSplittingICBSh` | 24/25 | 27.0 | 58.8 ± 44.4 | 1.230s | 0.1054 |
| `random-32-32-20` | 20 | `FAwareResidualICBSh` | 24/25 | 27.0 | 52.2 ± 32.2 | 1.043s | 0.1160 |
| `random-32-32-20` | 20 | `Learned_CL_ICBSh` | 24/25 | 26.0 | 47.7 ± 31.0 | 1.155s | — |
| `random-32-32-20` | 28 | `PlainCBS` | 4/25 | 10226.0 | 11491.0 ± 3533.7 | 60.121s | 0.0000 * |
| `random-32-32-20` | 28 | `PlainICBSh` | 12/25 | 140.0 | 215.3 ± 68.9 | 60.015s | 0.9306 |
| `random-32-32-20` | 28 | `RandomRankerICBSh` | 3/25 | 179.0 | 311.9 ± 135.2 | 60.093s | 0.4370 |
| `random-32-32-20` | 28 | `DisjointSplittingICBSh` | 18/25 | 194.0 | 262.8 ± 94.1 | 30.470s | 0.6617 |
| `random-32-32-20` | 28 | `FAwareResidualICBSh` | 18/25 | 194.0 | 238.9 ± 79.0 | 28.424s | 0.7186 |
| `random-32-32-20` | 28 | `Learned_CL_ICBSh` | 17/25 | 227.0 | 261.2 ± 79.5 | 19.637s | — |
| `room-32-32-4` | 16 | `PlainCBS` | 9/25 | 10885.0 | 11736.6 ± 4183.5 | 60.116s | 0.0000 * |
| `room-32-32-4` | 16 | `PlainICBSh` | 17/25 | 51.0 | 169.5 ± 79.7 | 12.336s | 0.0009 * |
| `room-32-32-4` | 16 | `RandomRankerICBSh` | 11/25 | 185.0 | 292.9 ± 116.0 | 60.013s | 0.0000 * |
| `room-32-32-4` | 16 | `DisjointSplittingICBSh` | 20/25 | 47.0 | 99.4 ± 58.4 | 7.387s | 0.0493 * |
| `room-32-32-4` | 16 | `FAwareResidualICBSh` | 21/25 | 44.0 | 75.9 ± 38.7 | 4.686s | 0.5347 |
| `room-32-32-4` | 16 | `Learned_CL_ICBSh` | 20/25 | 46.0 | 86.9 ± 47.6 | 4.845s | — |
| `room-32-32-4` | 20 | `PlainCBS` | 6/25 | 13815.0 | 10971.0 ± 2872.3 | 60.146s | 0.0000 * |
| `room-32-32-4` | 20 | `PlainICBSh` | 12/25 | 117.0 | 129.2 ± 44.9 | 60.018s | 0.0677 |
| `room-32-32-4` | 20 | `RandomRankerICBSh` | 6/25 | 194.0 | 227.8 ± 63.8 | 60.113s | 0.0002 * |
| `room-32-32-4` | 20 | `DisjointSplittingICBSh` | 16/25 | 54.0 | 121.2 ± 48.9 | 28.282s | 0.1878 |
| `room-32-32-4` | 20 | `FAwareResidualICBSh` | 18/25 | 56.0 | 90.6 ± 36.0 | 14.316s | 0.8387 |
| `room-32-32-4` | 20 | `Learned_CL_ICBSh` | 17/25 | 67.0 | 103.8 ± 42.4 | 12.728s | — |
