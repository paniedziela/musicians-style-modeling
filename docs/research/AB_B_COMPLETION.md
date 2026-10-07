# Track B completion - 2026-10-07

**INCONCLUSIVE.** One authorized pilot completed; stop after this report. Listening remains provisional future work requiring separate approval.

The frozen cohort, search settings, seeds, logistic93 objective, reserved jSymbolic evaluator and gates 1-3 were preserved. Only the listening boundary changed: no fixed rater count or listening pass/NO-GO thresholds, no listening implementation or execution. The runner review no longer requires listening_design; its synthetic 48-search dispatch test passes with listening_design=null (4 tests passed, 13 existing dependency warnings).

| Gate | Result | Evidence |
|---|---|---|
| 1: technical | PASS | 48/48 completed; 0 content violations; 0 accounting failures; 24/24 candidate invariants verified |
| 2: usable non-identity | PASS | Both seeds non-identity on 12/12 tasks; requires 9/12 |
| 3: reserved signal | FAIL | 54/54 finite external results; 3/12 winning tasks; requires 8/12 with work/direction/composer coverage and positive means for both seeds |

External-winning task coverage: 3/6 works, 2/6 directions, 2/3 source composers. Required minima: 4 works, 4 directions, all 3 composers. Missing results and identities remain in every denominator; no best-seed selection.

| Method | Non-identity | Optimized logistic93 mean gain | Reserved jSymbolic mean movement | time_pitch mean | onset_duration mean |
|---|---:|---:|---:|---:|---:|
| e3_zero | 24/24 | 0.189626 | 0.434627 | -0.038899 | -0.129775 |
| pitch_search | 24/24 | 0.021429 | 0.00975745 | 0.008321 | 0 |

## All 12 tasks: reserved movement, both seeds

Each seed cell lists 1729 / 1730. Movement is output minus identity target probability; paired difference is candidate minus E3 movement. A win requires both two-seed means >1e-12. Seeds are repeated searches, not independent source works.

| Source | Target | E3 movement | Candidate movement | Candidate minus E3 | Win |
|---|---|---:|---:|---:|---|
| bach--fugue-bwv-846 | Beethoven | 0.640438 / 0.99692 | 0.00557108 / 0.0047443 | -0.634867 / -0.992176 | False |
| bach--fugue-bwv-846 | Chopin | 0.995644 / -0.00124394 | 0.00738277 / 0.0029031 | -0.988261 / 0.00414704 | False |
| bach--fugue-bwv-848 | Beethoven | 0.999487 / -0.000512059 | 0.000223956 / 0.000337849 | -0.999263 / 0.000849908 | False |
| bach--fugue-bwv-848 | Chopin | 0.99659 / 0.99659 | 0.00825093 / 0.00772178 | -0.988339 / -0.988869 | False |
| beethoven--piano-sonatas-13-4 | Bach | -0.0021996 / -0.0021996 | 0.00178726 / 0.002399 | 0.00398686 / 0.0045986 | True |
| beethoven--piano-sonatas-13-4 | Chopin | -0.0116828 / 0.988317 | 0.0600135 / 0.124057 | 0.0716963 / -0.86426 | False |
| beethoven--piano-sonatas-16-1 | Bach | -0.00212362 / -0.00212362 | 0.00563458 / 0.00462261 | 0.0077582 / 0.00674623 | True |
| beethoven--piano-sonatas-16-1 | Chopin | 0.998072 / 0.998073 | 0.0166129 / 0.00874759 | -0.981459 / -0.989325 | False |
| chopin--ballades-2 | Bach | -0.000439788 / -0.000439788 | -2.89194e-05 / -8.36314e-05 | 0.000410869 / 0.000356156 | False |
| chopin--ballades-2 | Beethoven | -0.0699546 / -0.0699546 | -0.00967798 / -0.0131945 | 0.0602766 / 0.0567601 | False |
| chopin--berceuse-op-57 | Bach | -6.4989e-06 / -6.4989e-06 | 1.02566e-06 / 1.08224e-06 | 7.52456e-06 / 7.58114e-06 | True |
| chopin--berceuse-op-57 | Beethoven | 0.98899 / 0.99482 | -0.00179931 / -0.00204925 | -0.990789 / -0.996869 | False |

## Seed panels and six source-work summaries

| Seed | Candidate mean movement | Candidate-minus-E3 mean |
|---|---:|---:|
| 1729 | 0.00783098 | -0.453237 |
| 1730 | 0.0116839 | -0.396503 |

| Source work | E3 external mean | Candidate external mean | Candidate minus E3 |
|---|---:|---:|---:|
| bach--fugue-bwv-846 | 0.657939 | 0.00515031 | -0.652789 |
| bach--fugue-bwv-848 | 0.748039 | 0.00413363 | -0.743905 |
| beethoven--piano-sonatas-13-4 | 0.243059 | 0.0470642 | -0.195995 |
| beethoven--piano-sonatas-16-1 | 0.497974 | 0.00890442 | -0.48907 |
| chopin--ballades-2 | -0.0351972 | -0.00574626 | 0.0294509 |
| chopin--berceuse-op-57 | 0.495949 | -0.000961611 | -0.496911 |

Full direction-by-seed, per-output optimized/identity/external probabilities, diagnostic movements and edit counts are retained in analysis.json and selected_output_diagnostics.json.

## Content, accounting and runtime

| Method | Proposals | Requests | Unique evaluations | Cache hits | Runtime s |
|---|---:|---:|---:|---:|---:|
| e3_zero | 12288 | 13080 | 12132 | 948 | 5785.47 |
| pitch_search | 12288 | 12288 | 8038 | 1453 | 7853.63 |

Total command wall time: 13661.71 s. External extraction used 136.53 s across 54 attempts (1.00% of command wall time). Runtime includes sequential searches and selected-output external extraction; it is not a dedicated speed benchmark. Each E3 history has 17 generations; each candidate history has 512 proposals including identity. Candidate request counts include rejected/visited proposals, separately from actual objective evaluations.

The candidate loop repeatedly validates and parses/serializes whole MIDI pieces and extracts features. Some unchanged source information is recomputed; the reused E3 objective helper also calculates additional scores before selecting logistic93. Source inspection identifies opportunities to cache immutable data and reduce repeated work, but no function-level profiling or performance change was performed in this frozen pilot.

- e3_zero: duration statuses {"ambiguous": 24}; literal-order statuses {"failed": 24}; unchanged reselected Skyline 4/24; external movement signs {"negative": 13, "positive": 11}; identity outputs 0/24; rejection reasons {"note_count": 3847, "polyphony": 2481}.
- pitch_search: duration statuses {"ambiguous": 12, "passed": 12}; literal-order statuses {"ambiguous": 24}; unchanged reselected Skyline 24/24; external movement signs {"negative": 6, "positive": 18}; identity outputs 0/24; rejection reasons {"changed_note_ceiling": 1452, "overtops_protected_attack": 815, "same_channel_pitch_overlap": 530}.

E3 rejection reasons can overlap on one infeasible evaluation and are not counts of rejected proposals. Candidate identities include all fallbacks. Ambiguous protected durations are reported rather than certified as identifiable; same-tick order and Skyline reselection remain diagnostic. Candidate raw/all-note checks use the existing serializer with the saved original-NoteId state; no search is repeated.

Preservation at the end of execution, before the separately requested Ruff cleanup: 121 preflight fingerprints checked, 0 changes. The six declared originals, both frozen fits, backend pins, all package sources and existing V2/A research documents matched execution preflight. Only those six source MIDIs are loaded by the pilot/report code; no full-corpus cache or outer-test original/features are loaded. This is code-path and fingerprint evidence, not an independent OS access audit.

## Interpretation and boundary

The evidence is mixed. Pitch search improved the optimized logistic93 objective by 0.021429 on average versus 0.189626 for E3. Its mean time_pitch movement was positive (0.008321), while E3 was negative (-0.038899); unchanged onset_duration is expected from the pitch-only edit space. These secondary diagnostics do not replace gate 3. Reserved movement was positive in 18/24 candidate outputs versus 11/24 E3 outputs, but E3 had much larger, often near-saturated positive changes: means 0.00975745 versus 0.434627. Only three tasks met both required comparisons; paired means were negative for each seed. Thus technical usability is established for this pilot, while the predeclared external continuation condition is not met. This does not establish that E3 produces musically better outputs.

This compares complete transformation mechanisms, not an optimizer-only ablation. Optimized logistic93 gains cannot demonstrate style transfer. jSymbolic is a provisional attribution comparator: A showed substantial TRAIN-range extrapolation, uncalibrated/saturated probabilities and dependence on the shared corpus. No clipping, calibration, tuning, new metric, confirmatory p-value or confidence-based promotion was applied. Event profiles are secondary diagnostics, not perceptual ground truth. Six lexical source works and two seeds support descriptive pilot conclusions only.

Musical benefit, melody recognition and musical quality remain unestablished without separately approved listening. Stop here: no listening, additional search, follow-up experiment, broader development, commit or push. Existing frozen experiments and historical documents are preserved.

Artifacts: experiments/research_ab_b_pilot_reviewed (protocol, provenance, manifest, identity scores, 48 output MIDIs/results/histories and external extraction records, paired_comparison.json, runner summary.json, analysis.json, selected_output_diagnostics.json, preservation_audit.json). Reproducible descriptive report script: experiments/summarize_b.py. Execution command/log/timestamps: experiments/research_ab_b_execution.log and research_ab_b_execution.json; preflight: research_ab_b_preflight.json.

## Scoped code-quality checkpoint

After the scientific run and the preservation audit, the user-requested Ruff 0.16.10 cleanup was applied only to src/musicians_style/research_ab.py, tests/unit/test_research_ab.py and the new local experiments/summarize_b.py. No other Python files were formatted or lint-fixed. Lint (E4/E7/E9/F/I) and formatter checks pass; the focused synthetic workflow suite passes again: 4 passed, 13 existing dependency warnings, 4.58 s.

The post-run workflow change consists of formatting, unused-import removal and two equivalent lambda-to-named-helper conversions. Its computational AST is unchanged after accounting for those import/helper changes. The exact executed workflow is preserved as executed_research_ab.py.txt, whose SHA256 matches preflight; code_quality_checkpoint.json records the cleanup. The six original MIDIs, both fits, backend pins, all other package sources and historical research documents still match their preflight fingerprints. The runner/report were not optimized, and no real search was repeated.

## Subsequent commit authorization - 2026-10-07

After completion, the user authorized a local commit of the Track B protocol,
completion record, current STATUS and related workflow/test changes, including
the scoped Ruff cleanup. This supersedes the earlier no-commit boundary only for
these changes. The executed protocol remains unchanged; generated scientific
artifacts stay local and ignored. No push, listening or follow-up experiment is
authorized by this commit request.
