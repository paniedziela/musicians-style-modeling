# jSymbolic external evaluation of frozen E2/E3

150 accepted sources; 300 E2 and 300 E3 outputs. 300/300 source–target pairs scored in all comparisons; 0 extraction failures.

The existing 638-column jSymbolic 2.2 projection and logistic classifier are reused. Five models use the existing repeat-0 grouped outer folds; imputation, variance filtering and scaling are fitted on each training fold only. The held-out source and both outputs use the same model. Identity is the unchanged source score (movement zero).

Target movement is p(target|output) − p(target|source). Values below are equal-work means with the existing 2,000-draw work bootstrap 95% intervals (seed 1729); each work averages its source pieces and, overall, both targets. All comparisons use the same complete pairs. Row-weighted means are also retained in summary.json.

| Direction | Pairs | Works | E2 − identity | E3 − identity | E3 − E2 |
|---|---:|---:|---:|---:|---:|
| Overall | 300 | 87 | +0.2559 [+0.1946, +0.3192] | +0.2665 [+0.2038, +0.3244] | +0.01062 [-0.0423, +0.06394] |
| Bach→Beethoven | 59 | 30 | +0.3373 [+0.2177, +0.4653] | -0.01512 [-0.04579, +0.01139] | -0.3524 [-0.4735, -0.2422] |
| Bach→Chopin | 59 | 30 | +0.6279 [+0.5122, +0.7367] | +0.9399 [+0.8935, +0.9768] | +0.3121 [+0.1955, +0.4262] |
| Beethoven→Bach | 57 | 28 | -0.03928 [-0.07505, -0.01372] | -0.0391 [-0.07472, -0.01354] | +0.0001815 [-3.317e-06, +0.0004344] |
| Beethoven→Chopin | 57 | 28 | +0.556 [+0.4074, +0.7002] | +0.8364 [+0.7497, +0.9127] | +0.2805 [+0.1289, +0.4477] |
| Chopin→Bach | 34 | 29 | -0.058 [-0.1424, -0.005267] | -0.05796 [-0.1424, -0.005226] | +4.143e-05 [+1.582e-12, +0.0001242] |
| Chopin→Beethoven | 34 | 29 | +0.09604 [-0.08087, +0.2889] | -0.06947 [-0.231, +0.094] | -0.1655 [-0.3443, +0.01295] |

The overall E3−E2 interval includes zero; this evaluator does not establish an overall advantage for either method.

These are descriptive classifier-proxy results conditional on the fixed folds and fits, not evidence of authorship, musical quality or content preservation. Direction intervals are descriptive, without new tests or decision thresholds. The existing jSymbolic metadata sanitation also applies to these inputs.

Reproduce with the pinned project environment (Python 3.10, numpy 1.26.4, scikit-learn 1.5.1, mido 1.3.2) from the repository root:

```powershell
.\.venv\Scripts\python.exe tools/evaluate_e2_e3_jsymbolic.py --distribution "D:\Studia\inzynierka_dev\experiments\research_ab_verification_2026-10-05\jsymbolic\jSymbolic_2_2_user" --java "C:\Program Files\Java\jre1.8.0_291\bin\java.exe" --output "D:\Studia\inzynierka_dev\experiments\e2_e3_jsymbolic"
```

The extraction cache retains the pinned backend/runtime, feature schema and nullable values for all 750 inputs. Re-running reuses it and refits the five classifiers. Optional --source-cache reuses existing Track A source extractions. per_task.csv contains target probabilities and paired movements; fits.json records training/test IDs, parameters and fit warnings.
