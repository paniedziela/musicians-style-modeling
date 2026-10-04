# STATUS — Research V2 entry point

Updated **2026-10-04** after the separately authorized **V2-02 feature feasibility only**. V2-01 and V2-03 remain complete. **Stop for review after V2-02; local commits authorized, no push. Full E1c, V2-04 and V2-05 were not started.** Research V2 is an experiment phase, with no parallel package namespace.

## V2-02 completion

The predeclared nine repeat-0 / outer-fold-0 training samples each have two recorded extraction attempts for custom93 and isolated musif 1.2.4: 18/18 successful results per backend, with exact repeated schemas/values/outcomes. custom93 wraps the frozen implementation and matches all 150 frozen cache vectors exactly. musif provides 203-329 nullable numerical descriptors per sample (405-column pilot union); sample-dependent vocabulary and missing NumericTempo prevent treating this as a ready classifier matrix. No imputation, fitting or E1c classification ran.

See [V2_02_COMPLETION.md](research/V2_02_COMPLETION.md) for exact files, commands, environments, findings, limitations and commits. Definitive artifacts: `experiments/research_v2_02_2026-10-04/pilot_final`; dependencies are isolated under that task's `musif_env`, with a tracked full lock at `requirements/research-v2-02-musif.lock.txt`. The original `.venv` and frozen E1-E4 remain unchanged.

The V2-03 review clarification was recorded before feature implementation: ordering of distinct simultaneous same-tick MIDI events remains diagnostic and alone is not a hard violation of melodic identity. Protected absolute pitch, onset and note-off / identifiable duration remain the primary hard requirements. Completed V2-03 scientific artifacts, code and completion record are unchanged; their strict-order statistics retain their historical meaning.

## V2-03 completion

All 300 E2 outputs, 300 E3 outputs and 150 original source-self identity references were measured explicitly. The original source Skyline mask is retained. Observable protected on/off retention is separate from ambiguous voice duration/order; structural/technical checks and protected velocity have their own categories. Historical E3 uses its recorded allowed transposition and original same-tick order interpretation; the V2 exact-pitch measurement uses zero shift and strict observable event order. Frozen E1-E4 do not import the new modules.

See [V2_03_COMPLETION.md](research/V2_03_COMPLETION.md) for final findings, exact files, commands/results, limitations and local commit details. Scientific artifacts live in `experiments/research_v2_03_2026-10-04/content_verified`; run `tools/content_audit.py` with a fresh output directory to reproduce them. The pass began from clean `refactor/research-v2` at `fdf529f`, following V2-01 commits `d6d3330`, `2204c86`, `fdf529f`. The V2-01 snapshot below remains historical evidence.

## V2-01 recorded checkout and sources

| Source | Verified state |
|---|---|
| Active checkout | `D:/Studia/inzynierka_dev`, branch main; HEAD `324052a9cd5e8269fe406885bb9b65d86856ccdb`, commit dated 2026-10-04 19:16:29 +02:00 |
| GitHub main | `594e70190708808516b93731c397d42e55b34371`, dated 2026-09-22 12:27:55 UTC; rechecked read-only via GitHub connector on 2026-10-04 |
| Difference before this pass | One local documentation commit adds six Research V2 navigation/prompt documents; research code matches remote baseline |
| Current work | Uncommitted documentation, V2-01 audit modules/tool and test-tier support; exact change inventory in the local completion record |
| Selected literature | All 13 local `Literatura/selected/L*.pdf` sources read in the preceding audit; matrix records versions, SHA256 and all 19 required fields |
| ChatGPT project context | Named Inżynierka project context was unavailable; user instructions, local documents/results and connected repository are the available sources |

Current entry points are this file, RESEARCH_CONTEXT, PAPER_IMPLEMENTATION_MATRIX, EXPERIMENT_REGISTRY and ROADMAP. Historical audits/plans/results remain evidence; their stale statements about unfinished E3, all GAN work being pending, independent evaluator meaning or incomplete full-piece coverage are not current operational guidance. README now points here and describes actual setup/test tiers. Historical reports are unchanged.

## Verified frozen evidence

| Experiment | Status / findings |
|---|---|
| E1b | CLOSED / GO. 150 accepted samples, 87 work groups; Bach 59/30, Beethoven 57/28, Chopin 34/29. RF balanced accuracy 0.86424, clustered CI [0.83781, 0.89059], macro-F1 0.86956; retrained permutation p=0.01099. Frozen custom93 reproduces its cache in the preceding read-only audit. |
| E2 | CLOSED / FROZEN BASELINE. 300 outputs, recorded hashes match. Mean Δp_target 0.02767, CI [0.01986, 0.03623]. Historical global GA damages temporal content; preserve it for paired comparison. |
| E3 | CLOSED / GO WITH LIMITATIONS. 300 outputs, hashes match. Mean Δp_target 0.03771, CI [0.02850, 0.04604]; paired E3−E2 0.01004, CI [0.00134, 0.02240], p=0.0395. Objective/evaluator Pearson r=0.04395; 45 identity fallbacks. |
| E4.5 / E4.6 | CLOSED / validation NO-GO. E4.6: 86 validation outputs available; best.pt matches validation/status SHA256 e165ded23e0202dd6798dbad6ff16f5ca33fce8c6c13428983412e1fdf0c0d5d. Four validation gates fail; no outer-test artifacts found. Absence does not prove data was never inspected. |

These statistics come from frozen reports/results and the preceding audit, not new feature extraction/training/content calculations in V2-01. E1b demonstrates corpus discrimination rather than a universal style definition. RF is a **separate held-out evaluator** with shared-corpus/overlapping-feature dependence; event-profile measurements provide a more distinct perspective.

Historical E3 allowed ±6-semitone global transposition (233/300 outputs nonzero). It remains the transposition-allowed baseline. New transfer experiments protect original absolute melody pitches, onsets and note-offs / identifiable durations exactly. Distinct simultaneous same-tick ordering remains diagnostic under the review clarification above. Structural/technical invariants and unchanged velocity are reported separately. Overlapping events need semantic auditing; do not retroactively interpret old E3 under the new exact-pitch policy.

## Environment and local assets

Initial ambient import was `D:/Studia/inzynierka_dev/.venv/lib/site-packages/musicians_style/__init__.py`: inside the checkout but outside its source package. The thin audit entry point rejected it before research-module loading and wrote `experiments/research_v2_01_2026-10-04/ambient_import` diagnostics.

Editable setup succeeded using `.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .`; no dependency upgrades. Ambient import now resolves to **`D:/Studia/inzynierka_dev/src/musicians_style/__init__.py`**, search location `D:/Studia/inzynierka_dev/src/musicians_style`, direct_url editable=true. The guard checks resolved source/search locations under this checkout's `src/musicians_style`, not merely under checkout. Another worktree/stale/missing/unresolved package fails. Checkout-first PYTHONPATH is a documented fallback, not an automatic repair performed by the audit.

Environment: `.venv/Scripts/python.exe`, Python 3.10.20, Windows; recorded dependency versions include numpy 1.26.4, scipy 1.13.1, sklearn 1.5.1, torch 2.2.2+cu121, pytest 8.2.2 and Hypothesis 6.103.2. musif/music21/Partitura absent; Java available but jSymbolic JAR not found. V2-02 subsequently installed musif/music21 only in its isolated environment; this original project-environment snapshot remains valid.

Audit roots default to this checkout's datasets / experiments / Literatura; explicit roots take precedence over MSM_DATA_ROOT / MSM_RESULTS_ROOT / MSM_LITERATURE_ROOT. All resolved paths and selection sources/config fingerprints are recorded. Legacy loaders and directory layout are unchanged. Physical ASAP inventory: 235 score MIDI/XML pairs; canonical 150 all have matching XML. Matched notation supports a separate score-aware E1m investigation.

## V2-01 verification

All commands below used `.venv/Scripts/python.exe` from the active checkout. `-B` avoids bytecode files; `-p no:cacheprovider` avoids the inaccessible existing pytest cache.

| Command suffix | Result |
|---|---|
| `-B -m pytest -q -p no:cacheprovider --durations=10` | Fast default: 452 passed, 64 deselected, 14 existing dependency warnings; 19.59 s |
| `-B -m pytest -q -p no:cacheprovider -o addopts= tests/property --hypothesis-profile=dev --durations=5` | 17 passed, 13 existing warnings; 9.34 s |
| `-B -m pytest -q -p no:cacheprovider -o addopts= tests/unit tests/integration --durations=10` | Broad unit/regression/integration: 527 passed, 14 existing warnings; 42.89 s |
| `-B -m pytest -q -p no:cacheprovider tests/unit/test_asset_paths.py tests/unit/test_provenance.py tests/unit/test_research_audit.py tests/unit/test_property_profiles.py` | 23 V2-01 synthetic/subprocess/profile checks passed; initial focused run 3.51 s; final cleanup verification recorded in completion artifact |
| `-B tools/research_audit.py --scope inventory --output experiments/research_v2_01_2026-10-04/inventory_final` | Passed; 1.72 s wall time |
| `-B tools/research_audit.py --scope baseline --output experiments/research_v2_01_2026-10-04/baseline_final` | Passed; 9.85 s wall time; 24 passed / 2 unavailable historical-provenance checks / 0 failed |

Initial baseline (`baseline_before`, 7.06 s) and subsequent baseline fingerprints match for 1,004 evidence files, 34 frozen experiment source files, nine configurations and all 13 PDFs. The final review audit/report and completion record live in `experiments/research_v2_01_2026-10-04`; every audit uses a fresh child directory. Synthetic fixtures exercise root precedence/other checkout defaults, correct/stale/foreign/missing imports, unresolved source, missing assets/hash mismatches/conflicting snapshots/splits/checkpoint, output-dir rejection and no frozen-input mutation.

Default `python -m pytest` is fast unit/smoke only. Registered property/integration/regression/slow tiers keep expensive workflows without deleting assertions. Original property budgets remain 50/100/200/300 under full; dev caps each at 20. Real 150-source/600-output audits are artifacts outside pytest. The prior 493-unit run took 50.46 s; default now meets the below-30-s development target here without a wall-clock assertion.

Skipped deliberately: full-budget property execution, new feature/content extraction, training, output regeneration, outer E4 test and listening study. Profile budget retention is explicitly tested; full unit/integration/regression coverage ran. Historical provenance remains incomplete: E1 report/manifest commit references differ after metadata anonymization; E2 commit does not reconstruct a full environment; E3 lacks a complete recorded source/environment snapshot. Do not synthesize these missing facts.

## Next review and bounded backlog

V2-01, V2-03 and the separately authorized V2-02 are complete. Review V2-02 before further work. No full E1c classification, style-measure fitting or objective search was launched.

1. V2-01 — roots/provenance/read-only evidence audit/test tiers: COMPLETE.
2. V2-02 — nine-sample feature feasibility: COMPLETE, STOP FOR REVIEW.
3. V2-03 — 600-output content audit plus 150 identities: COMPLETE; artifacts preserved.
4. V2-04 — E1d competing held-out style measurements: planning only, MUST, MEDIUM.
5. V2-05 — objective pilot: provisional SHOULD, candidates/complexity/budget after E1d validity and another review.

ROADMAP retains the bounded backlog. Introduce capability modules only when authorized tasks need them. Do not start full E1c, V2-04 or V2-05, and do not push.
