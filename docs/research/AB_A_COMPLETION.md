# Track A completion record - 2026-10-06

Track A only is complete. The reviewed implementation was committed and pushed before execution. B was not run. No feature/model tuning.

| Measure | jSymbolic | custom93 |
|---|---:|---:|
| Work-balanced macro attribution | 0.952381 | 0.783069 |
| Descriptive work bootstrap 95% interval | [0.857143, 1.000000] | [0.641534, 0.917659] |
| Piece balanced accuracy | 0.966667 | 0.718981 |
| Training-retained dimensions | 457 / 638 | 93 / 93 |
| Positive / negative / null output movements | 8 / 10 / 0 | 13 / 5 / 0 |
| Mean target probability movement | 0.315402 | 0.031952 |

Same 76 training samples / 45 works; 44 validation samples / 24 works; 18 V2-05 outputs from three source works. All validation/output comparisons are matched. jSymbolic 138/138 extraction successes, zero missing/nonfinite fields; custom93 originals 120/120 and all 18 outputs scored. Zero fit warnings; 20 logistic iterations for external. Preprocessing and filtering use TRAIN only. The serialized fits precede scoring and exactly reproduce in-memory predictions.

The probability scales are not calibrated across backends: mean movements cannot establish which evaluator sees more musical improvement. Sign agreement is 9/18. Row Pearson r=0.172009, Spearman rho=-0.153289. Work-mean r=0.848804 / rho=1.0 is based on only THREE works and must not be read as robust agreement; its conditional bootstrap is degenerate (242/2000 draws undefined).

| Original V2-05 objective | jSymbolic positive / negative | custom93 positive / negative | Same sign |
|---|---:|---:|---:|
| rms67 | 3 / 3 | 3 / 3 | 4 / 6 |
| gaussian67 | 3 / 3 | 4 / 2 | 3 / 6 |
| logistic93 | 2 / 4 | 6 / 0 | 2 / 6 |

The new custom93 output probabilities reproduce saved V2-05 logistic93 probabilities exactly (maximum difference 0.0). All nine sign disagreements and six directions are retained in analysis.json/output_scores.json. No old fitted model was loaded.

Recommendation: freeze the exact current jSymbolic bundle as a PROVISIONAL external attribution comparator for B. Strong discrimination, complete coverage and informative disagreements justify this role. It is not a validated judge of musical style/quality or a sole B continuation gate. Outputs extrapolate substantially: max absolute TRAIN-standardized component ranges from 20.10 to 1429.67, often in rare melodic/tempo-standardized beat histogram bins. Large/saturated probability changes need this caveat; no clipping, calibration, regularization or descriptor change was applied. Listening and predeclared B continuation rules remain necessary; B remains unapproved.

Frozen bundle SHA256: `d25f17836e32ab2c963a7bfec34f0e84f7c5723831e6d1ee8c1d5a337e01d9a3`. Metadata SHA256: `05c33e82a057c28d0eb2acc0acd1045612ff29d22888faa665fcda27995ecbcb`. freeze_manifest.json fixes role, pins and limitations without authorizing B.

Read-only execution audit: 138 external attempts, 491.14 s; exactly 120 permitted originals and 18 V2-05 outputs; zero blocked requests/outer-test-original reads/full-cache reads; zero dataset writes or protected hash changes. The tree remained clean. No B run, merge, rebase or main change.

Artifacts: protocol.json, cohort.json, provenance.json, extractions.json, per-attempt XML/CSV/JSON/logs, fit.json, frozen_external.joblib, fit_manifest.json, real_scores.json, output_scores.json, summary.json, analysis.json, range_diagnostics.json, freeze_manifest.json, execution_audit.json.


## Commit / verification checkpoint

Reviewed implementation commits: `d3c9fa7` (A), `a3f6ac3` (B), `bb72b1c` (protocol/docs).
The clean branch was pushed to origin at `bb72b1c1d0bb69914b9af58bbeb03d039820488c`
before the one scientific A run. Final result documentation is a separate commit;
no scientific artifacts/model/data are tracked by Git.

Tests before execution: 602 passed / 73 deselected; standalone A 20 passed;
B/workflow/full property verification 30 passed, all synthetic. The extra A test
file isolates A checks for its logical commit. Original pinned config/schema
bytes and feature/classifier settings are unchanged. Git attributes prevent
cross-platform newline normalization from invalidating the original hashes.

Execution used the existing `.venv/Scripts/python.exe -B` and the same
`research_ab.evaluate_a` as the CLI, inside an independent audit wrapper:
`experiments/research_ab_a_review_2026-10-05/run_a_guarded.py`.
The review file was explicitly approved for A by this user's request after push;
B's approval remains false. Detailed guarded execution/log/authorization are in
`experiments/research_ab_a_review_2026-10-05`; definitive A results are in
`experiments/research_ab_a_evaluation_reviewed`.

Run started 2026-10-05 and completed 2026-10-06 (Europe/Warsaw). The output path
is the predeclared fresh path, not renamed after execution. Fixed r00/f00/inner00
partition; no whole-corpus/outer evaluation or re-fit/tuning after results.
The local main ref remains `594e70190708808516b93731c397d42e55b34371`.

Stop for A review. The provisional frozen attribution evaluator does not
constitute authorization for the B pilot or a claim of perceptual validity.

Command-local `PYTHONPATH` selected the reviewed checkout source. No project
dependencies, Java installation, feature/model settings or GPU configuration
were changed. Final saved score finiteness, pin/model/metadata SHA256 and B
approval-false checks all pass.
