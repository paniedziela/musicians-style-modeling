# STATUS — Research V2 entry point

Updated **2026-10-04** after authorized documentation + V2-01 implementation. **Stop for review: no commit/push, no V2-02 launch.** The earlier Plan Mode limitation is superseded; files have now changed. Research V2 is an experiment phase, with no parallel package namespace.

## Checkout and sources

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

Historical E3 allowed ±6-semitone global transposition (233/300 outputs nonzero). It remains the transposition-allowed baseline. New transfer experiments protect original absolute melody pitches, onsets, note-offs/durations and event order exactly. Structural/technical invariants and unchanged velocity are reported separately. Overlapping events need semantic auditing; do not retroactively interpret old E3 under the new exact-pitch policy.

## Environment and local assets

Initial ambient import was `D:/Studia/inzynierka_dev/.venv/lib/site-packages/musicians_style/__init__.py`: inside the checkout but outside its source package. The thin audit entry point rejected it before research-module loading and wrote `experiments/research_v2_01_2026-10-04/ambient_import` diagnostics.

Editable setup succeeded using `.venv/Scripts/python.exe -m pip install --no-deps --no-build-isolation -e .`; no dependency upgrades. Ambient import now resolves to **`D:/Studia/inzynierka_dev/src/musicians_style/__init__.py`**, search location `D:/Studia/inzynierka_dev/src/musicians_style`, direct_url editable=true. The guard checks resolved source/search locations under this checkout's `src/musicians_style`, not merely under checkout. Another worktree/stale/missing/unresolved package fails. Checkout-first PYTHONPATH is a documented fallback, not an automatic repair performed by the audit.

Environment: `.venv/Scripts/python.exe`, Python 3.10.20, Windows; recorded dependency versions include numpy 1.26.4, scipy 1.13.1, sklearn 1.5.1, torch 2.2.2+cu121, pytest 8.2.2 and Hypothesis 6.103.2. musif/music21/Partitura absent; Java available but jSymbolic JAR not found. Their installation/extraction belongs to separately reviewed V2-02.

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

V2-01 is complete after final verification. Review its diff, setup choice, tier organization and artifacts before further work. No deviations in scientific scope; the explicit editable environment change is recorded, and full properties were deferred as the revised verification plan permits.

1. V2-01 — roots/provenance/read-only evidence audit/test tiers: this pass only, MUST, MEDIUM.
2. V2-02 — nine-sample feature feasibility: planning only, MUST, MEDIUM.
3. V2-03 — 600-output content audit: planning only, MUST, MEDIUM.
4. V2-04 — E1d competing held-out style measurements: planning only, MUST, MEDIUM.
5. V2-05 — objective pilot: provisional SHOULD, candidates/complexity/budget after E1d validity and another review.

ROADMAP gives exact files, acceptance tests, artifacts, agent passes, compute cost, blocking dependencies and parallelism. Introduce capability modules only when those tasks need them. Do not start V2-02–V2-05, commit or push in this pass.
