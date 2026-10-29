# Phase 3 Statistical Benchmark Summary (25 Scenarios/Cell, 7 Solvers)

## All 25 Scenarios (scen_ids 1–25)

| Map | k | Solver | Solve Rate | Median Nodes | Mean ± 95% CI Nodes | Median A* Calls | Median Runtime (s) | Median Peak Mem (KB) | Wilcoxon p vs Learned_XGB_CL |
|---|---|---|---|---|---|---|---|---|---|
| `random-32-32-20` | 20 | `PlainCBS` | 10/25 | 350.0 | 230.8 ± 55.1 | 86 | 0.495s | 818.2 | 0.0000 * |
| `random-32-32-20` | 20 | `PlainICBSh` | 20/25 | 19.0 | 34.2 ± 13.7 | 57 | 0.346s | 416.6 | 0.1834 |
| `random-32-32-20` | 20 | `RandomRankerICBSh` | 24/25 | 20.0 | 35.5 ± 14.4 | 22 | 0.216s | 113.5 | 0.0186 * |
| `random-32-32-20` | 20 | `DisjointSplittingICBSh` | 23/25 | 18.0 | 33.4 ± 14.8 | 20 | 0.195s | 103.7 | 0.5000 |
| `random-32-32-20` | 20 | `FAwareResidualICBSh` | 23/25 | 18.0 | 33.4 ± 14.8 | 20 | 0.153s | 76.0 | 0.5000 |
| `random-32-32-20` | 20 | `Learned_Linear_CL_ICBSh` | 23/25 | 18.0 | 33.0 ± 14.7 | 20 | 0.261s | 463.8 | 0.4461 |
| `random-32-32-20` | 20 | `Learned_XGB_CL_ICBSh` | 24/25 | 18.0 | 32.3 ± 13.0 | 20 | 0.281s | 490.3 | — |
| `random-32-32-20` | 28 | `PlainCBS` | 1/25 | 350.0 | 340.7 ± 18.2 | 105 | 0.755s | 885.6 | 0.0000 * |
| `random-32-32-20` | 28 | `PlainICBSh` | 11/25 | 52.0 | 56.4 ± 12.2 | 118 | 2.502s | 984.1 | 0.6617 |
| `random-32-32-20` | 28 | `RandomRankerICBSh` | 14/25 | 76.0 | 70.6 ± 13.9 | 67 | 2.151s | 801.3 | 0.0351 * |
| `random-32-32-20` | 28 | `DisjointSplittingICBSh` | 18/25 | 53.0 | 66.7 ± 16.5 | 42 | 1.061s | 259.9 | 0.1112 |
| `random-32-32-20` | 28 | `FAwareResidualICBSh` | 20/25 | 53.0 | 68.2 ± 17.3 | 28 | 1.035s | 193.5 | 0.0083 * |
| `random-32-32-20` | 28 | `Learned_Linear_CL_ICBSh` | 19/25 | 67.0 | 67.3 ± 15.7 | 34 | 1.234s | 1065.6 | 0.0147 * |
| `random-32-32-20` | 28 | `Learned_XGB_CL_ICBSh` | 20/25 | 55.0 | 60.7 ± 14.5 | 28 | 1.262s | 1119.2 | — |
| `room-32-32-4` | 16 | `PlainCBS` | 10/25 | 350.0 | 248.3 ± 54.7 | 111 | 0.484s | 910.8 | 0.0000 * |
| `room-32-32-4` | 16 | `PlainICBSh` | 12/25 | 32.0 | 71.4 ± 24.7 | 76 | 1.428s | 1016.3 | 0.1671 |
| `room-32-32-4` | 16 | `RandomRankerICBSh` | 13/25 | 38.0 | 66.6 ± 23.8 | 59 | 1.141s | 911.7 | 0.2048 |
| `room-32-32-4` | 16 | `DisjointSplittingICBSh` | 14/25 | 26.0 | 56.6 ± 22.9 | 28 | 0.566s | 625.0 | 0.8448 |
| `room-32-32-4` | 16 | `FAwareResidualICBSh` | 16/25 | 26.0 | 52.7 ± 21.2 | 16 | 0.340s | 196.8 | 0.9884 |
| `room-32-32-4` | 16 | `Learned_Linear_CL_ICBSh` | 14/25 | 27.0 | 63.1 ± 24.5 | 17 | 0.388s | 701.2 | 0.4861 |
| `room-32-32-4` | 16 | `Learned_XGB_CL_ICBSh` | 15/25 | 28.0 | 64.0 ± 23.4 | 18 | 0.394s | 743.4 | — |
| `room-32-32-4` | 20 | `PlainCBS` | 4/25 | 350.0 | 306.0 ± 37.0 | 111 | 0.573s | 914.8 | 0.0000 * |
| `room-32-32-4` | 20 | `PlainICBSh` | 8/25 | 71.0 | 75.0 ± 19.3 | 147 | 2.504s | 1240.4 | 0.5814 |
| `room-32-32-4` | 20 | `RandomRankerICBSh` | 9/25 | 46.0 | 69.4 ± 19.8 | 89 | 2.505s | 1115.4 | 0.9152 |
| `room-32-32-4` | 20 | `DisjointSplittingICBSh` | 12/25 | 50.0 | 59.6 ± 17.9 | 42 | 1.865s | 835.9 | 0.9803 |
| `room-32-32-4` | 20 | `FAwareResidualICBSh` | 14/25 | 38.0 | 57.8 ± 18.2 | 34 | 1.271s | 687.0 | 0.9987 |
| `room-32-32-4` | 20 | `Learned_Linear_CL_ICBSh` | 12/25 | 54.0 | 70.2 ± 20.2 | 36 | 1.868s | 1325.7 | 0.8672 |
| `room-32-32-4` | 20 | `Learned_XGB_CL_ICBSh` | 12/25 | 57.0 | 76.2 ± 21.0 | 36 | 1.843s | 1377.1 | — |
| `warehouse-32-32-aisles` | 12 | `PlainCBS` | 24/25 | 6.0 | 33.2 ± 28.2 | 20 | 0.022s | 69.2 | 0.0000 * |
| `warehouse-32-32-aisles` | 12 | `PlainICBSh` | 24/25 | 4.0 | 11.4 ± 11.5 | 14 | 0.017s | 41.3 | 0.8413 |
| `warehouse-32-32-aisles` | 12 | `RandomRankerICBSh` | 24/25 | 4.0 | 11.5 ± 11.5 | 12 | 0.012s | 19.8 | 0.1587 |
| `warehouse-32-32-aisles` | 12 | `DisjointSplittingICBSh` | 24/25 | 4.0 | 11.4 ± 11.5 | 12 | 0.012s | 19.7 | — |
| `warehouse-32-32-aisles` | 12 | `FAwareResidualICBSh` | 24/25 | 4.0 | 11.4 ± 11.5 | 12 | 0.011s | 19.7 | — |
| `warehouse-32-32-aisles` | 12 | `Learned_Linear_CL_ICBSh` | 24/25 | 4.0 | 11.4 ± 11.5 | 12 | 0.012s | 19.7 | — |
| `warehouse-32-32-aisles` | 12 | `Learned_XGB_CL_ICBSh` | 24/25 | 4.0 | 11.4 ± 11.5 | 12 | 0.012s | 19.7 | — |
| `warehouse-32-32-aisles` | 16 | `PlainCBS` | 20/25 | 37.0 | 109.2 ± 53.9 | 44 | 0.069s | 151.2 | 0.0000 * |
| `warehouse-32-32-aisles` | 16 | `PlainICBSh` | 24/25 | 7.0 | 16.0 ± 11.4 | 21 | 0.047s | 74.8 | 0.0899 |
| `warehouse-32-32-aisles` | 16 | `RandomRankerICBSh` | 24/25 | 7.0 | 15.6 ± 11.4 | 16 | 0.031s | 37.0 | 0.3527 |
| `warehouse-32-32-aisles` | 16 | `DisjointSplittingICBSh` | 24/25 | 7.0 | 15.5 ± 11.3 | 16 | 0.031s | 36.8 | — |
| `warehouse-32-32-aisles` | 16 | `FAwareResidualICBSh` | 24/25 | 7.0 | 15.5 ± 11.3 | 16 | 0.032s | 36.8 | — |
| `warehouse-32-32-aisles` | 16 | `Learned_Linear_CL_ICBSh` | 24/25 | 7.0 | 16.0 ± 11.4 | 16 | 0.046s | 39.9 | 0.1587 |
| `warehouse-32-32-aisles` | 16 | `Learned_XGB_CL_ICBSh` | 24/25 | 7.0 | 15.5 ± 11.3 | 16 | 0.058s | 39.9 | — |

---

## Held-Out Test Scenarios (scen_ids 16–25)

| Map | k | Solver | Solve Rate | Median Nodes | Mean ± 95% CI Nodes | Median A* Calls | Median Runtime (s) | Median Peak Mem (KB) | Wilcoxon p vs Learned_XGB_CL |
|---|---|---|---|---|---|---|---|---|---|
| `random-32-32-20` | 20 | `PlainCBS` | 4/10 | 350.0 | 244.2 ± 86.9 | 93 | 0.490s | 781.7 | 0.0010 * |
| `random-32-32-20` | 20 | `PlainICBSh` | 6/10 | 21.0 | 45.0 ± 30.1 | 61 | 0.414s | 747.3 | 0.7812 |
| `random-32-32-20` | 20 | `RandomRankerICBSh` | 9/10 | 38.5 | 53.5 ± 31.3 | 64 | 0.610s | 525.7 | 0.0156 * |
| `random-32-32-20` | 20 | `DisjointSplittingICBSh` | 8/10 | 29.5 | 50.2 ± 33.6 | 28 | 0.234s | 132.7 | 0.5000 |
| `random-32-32-20` | 20 | `FAwareResidualICBSh` | 8/10 | 29.5 | 50.2 ± 33.6 | 20 | 0.290s | 114.1 | 0.5000 |
| `random-32-32-20` | 20 | `Learned_Linear_CL_ICBSh` | 8/10 | 28.0 | 50.3 ± 33.7 | 20 | 0.331s | 611.5 | 0.3125 |
| `random-32-32-20` | 20 | `Learned_XGB_CL_ICBSh` | 9/10 | 32.0 | 46.5 ± 28.9 | 20 | 0.394s | 620.2 | — |
| `random-32-32-20` | 28 | `PlainCBS` | 1/10 | 350.0 | 326.8 ± 45.5 | 98 | 0.631s | 874.9 | 0.0010 * |
| `random-32-32-20` | 28 | `PlainICBSh` | 5/10 | 59.0 | 63.7 ± 21.9 | 164 | 2.392s | 977.2 | 0.7930 |
| `random-32-32-20` | 28 | `RandomRankerICBSh` | 6/10 | 83.0 | 80.7 ± 24.9 | 76 | 2.266s | 845.6 | 0.1094 |
| `random-32-32-20` | 28 | `DisjointSplittingICBSh` | 6/10 | 65.5 | 81.8 ± 30.7 | 46 | 1.334s | 259.2 | 0.0195 * |
| `random-32-32-20` | 28 | `FAwareResidualICBSh` | 8/10 | 65.5 | 76.1 ± 27.9 | 28 | 1.222s | 210.2 | 0.0352 * |
| `random-32-32-20` | 28 | `Learned_Linear_CL_ICBSh` | 7/10 | 69.5 | 74.0 ± 25.8 | 35 | 1.312s | 1100.6 | 0.0684 |
| `random-32-32-20` | 28 | `Learned_XGB_CL_ICBSh` | 7/10 | 57.5 | 69.1 ± 24.6 | 28 | 1.274s | 1149.8 | — |
| `room-32-32-4` | 16 | `PlainCBS` | 2/10 | 350.0 | 285.4 ± 84.4 | 160 | 0.530s | 976.9 | 0.0010 * |
| `room-32-32-4` | 16 | `PlainICBSh` | 2/10 | 150.0 | 109.0 ± 37.1 | 142 | 1.747s | 1023.5 | 0.2188 |
| `room-32-32-4` | 16 | `RandomRankerICBSh` | 3/10 | 141.0 | 103.5 ± 37.5 | 112 | 1.880s | 975.1 | 0.4219 |
| `room-32-32-4` | 16 | `DisjointSplittingICBSh` | 4/10 | 100.5 | 86.7 ± 40.2 | 48 | 1.471s | 799.1 | 0.7500 |
| `room-32-32-4` | 16 | `FAwareResidualICBSh` | 5/10 | 86.5 | 77.9 ± 38.0 | 20 | 1.003s | 548.7 | 0.8906 |
| `room-32-32-4` | 16 | `Learned_Linear_CL_ICBSh` | 4/10 | 129.5 | 93.8 ± 40.4 | 28 | 1.308s | 1258.7 | 0.5938 |
| `room-32-32-4` | 16 | `Learned_XGB_CL_ICBSh` | 4/10 | 119.5 | 94.2 ± 37.1 | 38 | 1.768s | 1359.2 | — |
| `room-32-32-4` | 20 | `PlainCBS` | 2/10 | 350.0 | 290.0 ± 78.4 | 117 | 0.376s | 848.3 | 0.0010 * |
| `room-32-32-4` | 20 | `PlainICBSh` | 2/10 | 96.5 | 93.4 ± 34.2 | 145 | 2.286s | 1329.4 | 0.7812 |
| `room-32-32-4` | 20 | `RandomRankerICBSh` | 2/10 | 56.5 | 75.7 ± 36.7 | 102 | 2.506s | 1158.6 | 1.0000 |
| `room-32-32-4` | 20 | `DisjointSplittingICBSh` | 2/10 | 47.5 | 68.5 ± 34.6 | 72 | 2.550s | 981.3 | 0.9609 |
| `room-32-32-4` | 20 | `FAwareResidualICBSh` | 3/10 | 38.0 | 67.7 ± 37.7 | 68 | 2.237s | 1153.6 | 0.9922 |
| `room-32-32-4` | 20 | `Learned_Linear_CL_ICBSh` | 2/10 | 75.5 | 79.3 ± 35.4 | 34 | 2.253s | 1753.8 | 1.0000 |
| `room-32-32-4` | 20 | `Learned_XGB_CL_ICBSh` | 2/10 | 119.0 | 96.3 ± 36.9 | 38 | 2.454s | 1682.6 | — |
| `warehouse-32-32-aisles` | 12 | `PlainCBS` | 10/10 | 5.0 | 21.7 ± 21.9 | 18 | 0.014s | 54.5 | 0.0039 * |
| `warehouse-32-32-aisles` | 12 | `PlainICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.010s | 29.2 | — |
| `warehouse-32-32-aisles` | 12 | `RandomRankerICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.007s | 19.4 | — |
| `warehouse-32-32-aisles` | 12 | `DisjointSplittingICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.007s | 19.4 | — |
| `warehouse-32-32-aisles` | 12 | `FAwareResidualICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.007s | 19.4 | — |
| `warehouse-32-32-aisles` | 12 | `Learned_Linear_CL_ICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.007s | 19.4 | — |
| `warehouse-32-32-aisles` | 12 | `Learned_XGB_CL_ICBSh` | 10/10 | 2.5 | 5.6 ± 4.2 | 12 | 0.008s | 19.4 | — |
| `warehouse-32-32-aisles` | 16 | `PlainCBS` | 9/10 | 19.5 | 73.3 ± 68.7 | 30 | 0.035s | 110.7 | 0.0020 * |
| `warehouse-32-32-aisles` | 16 | `PlainICBSh` | 10/10 | 5.0 | 8.8 ± 5.6 | 18 | 0.022s | 44.1 | 0.5000 |
| `warehouse-32-32-aisles` | 16 | `RandomRankerICBSh` | 10/10 | 5.0 | 9.0 ± 5.7 | 16 | 0.020s | 22.7 | 0.2500 |
| `warehouse-32-32-aisles` | 16 | `DisjointSplittingICBSh` | 10/10 | 5.0 | 8.6 ± 5.3 | 16 | 0.021s | 22.7 | — |
| `warehouse-32-32-aisles` | 16 | `FAwareResidualICBSh` | 10/10 | 5.0 | 8.6 ± 5.3 | 16 | 0.021s | 22.7 | — |
| `warehouse-32-32-aisles` | 16 | `Learned_Linear_CL_ICBSh` | 10/10 | 5.0 | 8.6 ± 5.3 | 16 | 0.021s | 22.7 | — |
| `warehouse-32-32-aisles` | 16 | `Learned_XGB_CL_ICBSh` | 10/10 | 5.0 | 8.6 ± 5.3 | 16 | 0.021s | 22.7 | — |
