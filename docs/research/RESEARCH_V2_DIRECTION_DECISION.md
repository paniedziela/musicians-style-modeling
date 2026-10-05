# Research V2 consolidation and direction decision

Date: 2026-10-05. Audited tip: `6895b0c`, branch `refactor/research-v2`.

**Decision: pursue two bounded tracks: (A) a minimal jSymbolic external-feature check, and (B) ONE transfer candidate: L0103-inspired accompaniment-only local pitch search.** Track A tests dependence on custom features; Track B investigates a materially different transformation mechanism. jSymbolic evaluation is supporting evidence, not the whole next implementation. No current evaluator is musical ground truth. This document supersedes the ambiguous future-action guidance in STATUS, ROADMAP and RESEARCH_CONTEXT. Historical protocols, completion records, measurements and code remain unchanged. This recommendation does not authorize a new study.

## 1. Baseline and plan reconstruction

Commit `324052a` introduced these five baseline files under `docs/research`:

| Document | Intended role |
|---|---|
| [MASTER_PROMPT_WORK.md](MASTER_PROMPT_WORK.md) | Research brief: inspect evidence/literature, challenge architecture, compare MIR tools, separate content/style, audit before implementation. |
| [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md) | Task, operational definitions, research questions, leakage rules and non-goals. |
| [PAPER_IMPLEMENTATION_MATRIX.md](PAPER_IMPLEMENTATION_MATRIX.md) | Papers mapped to reproducible/adaptable components and priorities, not mandatory implementations. |
| [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) | Frozen E1-E4 identities versus proposed experiments, hypotheses, protocols and artifacts. |
| [ROADMAP.md](ROADMAP.md) | Dependency order, implementation gates, bounded scope and deferred work. |

Counting distinction: [docs/STATUS.md](../STATUS.md) was the sixth Markdown file in that commit and the operational entry point. MASTER's *five deliverables* are STATUS plus the other four research documents, excluding MASTER itself. The table lists the five original research-reference files; neither inventory should erase STATUS or MASTER. Use `324052a` versions to reconstruct original intent, not today's edited navigation documents.

**Before V2-01:** freeze/orient and external roots; introduce narrow backend/schema/style/content capabilities; perform E1c comparing custom93 with musif, jSymbolic and selected music21; then E1d comparing RMS, classifier, Gaussian, event profiles and feasible StyleRank; then classifier/Gaussian objective variants and equal-budget E3 comparison. E1d requested at least two demonstrably usable independent measures. XML and a selected neural comparison were later options. Absolute-pitch/onset protection was not yet the fully specified V2 contract: melody identity/order were hard, rhythmic preservation experiment-dependent. The matrix favored adapting event-profile principles over copying groove-specific bins.

**Changes during V2, in actual commit order:**

- V2-01 (`d6d3330`, `2204c86`, `fdf529f`) added provenance/import audits and test tiers, replacing broad phases with five bounded tasks. Exact pitch/onset/off/order, nine lexical training samples and explicit review/eligibility gates became concrete later specifications.
- V2-03 (`df2f933`, `f12002b`) preceded V2-02. Overlap/FIFO and Skyline evidence supported separate observable identity, identifiable duration and ordering assessments. Before V2-02, review made distinct simultaneous same-tick order diagnostic. This is an explicit policy clarification informed by ambiguity, not a retroactive rewrite of V2-03.
- V2-02 (`88e837f`, `fb1bc06`) established musif feasibility but exposed missing tempo/varying vocabulary. The numeric projection and isolated environment were implementation adaptations. Full E1c remained unimplemented.
- V2-04 (`a2a1d16`, `df83958`, `5d9cbab`) implemented five measures. The Gaussian formula existed originally; its floor, logistic settings/solver, event bins/overflow/all-note aggregation and bootstrap rules were later specifications. Serialized RMS evidence justified removing an unintended execution-failure gate while retaining discrepancies; formulas did not change.
- V2-05 (`cc1051a`, `6895b0c`) followed explicit user selection of three objectives, superseding V2-01's provisional up-to-two eligible candidates. Cohort, budget, adapter and time_pitch's primary designation were later protocol choices. Better V2-04 ranking motivated interest in time_pitch; it did not validate that priority musically.

**Drift: yes, in priority and evidential strength.** E1c/external MIR and StyleRank were postponed while objective variants proceeded on custom93/67 and adapted histograms. The bounded roadmap allowed V2-04 without external features; V2-05 was explicitly authorized. This was documented adaptation, not unauthorized expansion. However, independent, demonstrably usable style evaluation remains only partly established. Administrative completion and above-chance ranking do not close that gap. Policy/numerical choices must not be described as evidence-driven discoveries or original commitments.

**Evidence-driven adaptations:** import/provenance guards and test tiers addressed observed environment/timing problems; raw-event separation addressed overlap/reselection evidence; musif's nullable schema addressed actual extraction outcomes; RMS discrepancy reporting preserved observed serialization effects. V2-04/V2-05 justify treating objective/evaluator disagreement as unresolved.

**Ad hoc research-process choices (subsequently predeclared):** exact absolute-pitch policy, simultaneous-order policy, nine-sample lexical selection, Gaussian floor, logistic settings, fixed four-beat/all-note profiles with overflow handling, pilot's three objectives/cohort/budget and time_pitch priority. Some were explicit user instructions, others routine engineering judgments. They were not all arbitrary or output-tuned, but results do not establish them as musically optimal. Commit history cannot recover undocumented motivations or every intervening conversation; attribution above follows the recorded protocols/completions.

## 2. Stage audit

“Still needed” means retain the capability/evidence, not repeat completed runs. Runner/tests accompany the named modules.

| Stage | Question answered | Useful result | Unresolved issue | Code added | Still needed? yes/no |
|---|---|---|---|---|---|
| V2-01 | Can checkout/evidence be identified and safely audited? | Roots/import guards, provenance; 24 passed checks, two historical gaps; test tiers. | Missing historical environments cannot be reconstructed. | `asset_paths`, `provenance`, `research_audit`; tier support. | Yes |
| V2-02 | Is external MIDI extraction reproducible? | 9/9 musif inputs twice; exact 150-row custom93 cache audit. | 203-329 columns/input, missing tempo; no external discrimination study. | `feature_backends/custom93`, `musif`, `feature_feasibility`; worker/lock. | Yes |
| V2-03 | What do frozen outputs preserve? | 300/300 E3 observable events under historical transposition; 273 ambiguous durations; 19 reselection discrepancies. | Skyline/voice lineage approximate; soft harmony/phrase analysis absent. | `content_metrics`, `content_audit`. | Yes |
| V2-04 | Do style definitions discriminate composers and agree? | Fold-local scoring/ranking, disagreement, 13 serialized RMS discrepancies. | No external representation/perceptual validation; inspected folds exploratory. | `style_metrics`, `style_audit`. | Yes |
| V2-05 | Can three objectives share bounded exact-pitch search? | 18 feasible outputs with optimized gains; disagreements documented. | Three works, all durations ambiguous; no validated winner/transfer success. | `objective_pilot`, external E3 adapter. | No: completed runner; retain adapter/evidence |

Sources: [STATUS](../STATUS.md), [V2-02](V2_02_COMPLETION.md), [V2-03](V2_03_COMPLETION.md), [V2-04](V2_04_COMPLETION.md), [V2-05](V2_05_COMPLETION.md), their protocols and corresponding source/tests. All statistics below are previously recorded results, not recomputations.

## 3. Evaluator validity: four different claims

| Evaluator | Implementation correctness evidence | Composer discrimination here | Independence from optimized objectives | Musical/perceptual validity |
|---|---|---|---|---|
| RMS67 | Frozen-formula tests; serialization differences retained. | Work-balanced ranking .5542, CI [.5009,.6103]. | Optimized by historical E3 and its pilot; shared Gaussian representation. | No corpus listener validation; marginal distributions omit much musical structure. |
| Gaussian67 | Formula/floor/train-only tests and saved-fit equivalence. | .5582 [.5082,.6107]. | Optimized in its pilot; same 67 components as RMS, both link and scale differ. | No corpus listener validation; satisfaction is not likelihood or demonstrated style similarity. |
| Logistic93 / RF | Leakage/pipeline/serialization checks; separate E1b validation. | Logistic .8237 [.7689,.8774]; E1b RF BA .86424 with permutation evidence; different protocols/statistics. | Logistic circular when optimized; separate RF shares custom93/corpus. | Attribution evidence, not validation of convincing edited music. |
| time_pitch | Bin/lag/overflow/symmetry/transposition/empty/tie tests; deterministic saved scores. No documented upstream parity test. | .6980 [.6290,.7660]; useful exploratory signal across 87 works, not 750 independent examples. | Unoptimized in V2-05; distinct representation but shared notes/parser/corpus and interval/rhythm information. | No validation that movement measures composer resemblance in transformed piano music. |
| onset_duration | Synthetic bin/overflow/identity and saved-fit tests; no documented upstream parity test. | .3979 [.3371,.4568], marginally above 1/3 under this descriptive CI. | Unoptimized diagnostic; overlaps objective rhythm/duration and shares parser/corpus. | No listener calibration; weak discrimination evidence. |

V2-04 bootstrap is conditional on fitted folds, without refitting or multiplicity correction. Historically inspected folds cannot become a fresh confirmatory test. Composer may be confounded with repertoire/form/meter/edition. Real-piece ranking does not validate an ordinal score on edited out-of-distribution music.

`time_pitch` pools piano-note pairs within four quarter beats: it loses absolute key/register, duration, velocity, voice/phrase identity and long-range structure; density changes pair weighting. `onset_duration` uses onset modulo four in every meter and clips at two beats, inheriting FIFO ambiguity. V2-04 reports non-4/4 in 117/150 sources, meter changes in 65, and 2,112,621/15,073,453 pair observations excluded for pitch overflow. Both mix protected melody with editable accompaniment: fixed content might drive discrimination. That is a plausible confound, not a measured explanation.

E3 mean time_pitch movement .003305 has CI [-.001711,.009265]; onset_duration .011645 has CI [-.001235,.024345]. V2-05 improves time_pitch in 3/18 outputs and onset_duration in 0/18. **This establishes disagreement and absent demonstrated success, not proof of musical failure or evaluator correctness.** Do not optimize diagnostics merely to turn their scores positive, tune bins on pilot outcomes, or select a winner from three works.

## 4. Literature-to-implementation mapping

Relevant stored PDFs were checked directly, including Groove2Groove V/VI-D and StyleRank 5-7; the [matrix](PAPER_IMPLEMENTATION_MATRIX.md) remains the full paper/version index.

| Literature | Current relationship and missing contribution |
|---|---|
| Herremans L0034 / jSymbolic-jMIR | jSymbolic -> composer classifier -> probability-guided VNS. V2 logistic/GA borrows the principle but uses custom93/different edits. Original fixed rhythm/counterpoint differs from exact melody transfer. Classification accuracy alone cannot justify classifier-only generation. |
| Groove2Groove L0074 | Explicit harmonic/chord content versus accompaniment style, chroma preservation versus profile cosine. VI-D/Fig.8 validates song/genre similarities on Bodhidharma, not composer transfer/listeners here. V2 merges notes, changes reference weighting and clips long durations where the paper excludes overflow. Its full content/style separation is not reproduced. |
| StyleRank L0230 | Categorical chord/interval/voice-leading/n-gram features and RF leaf similarity offer another scorer. BachBot human-ranking comparison: 5,967 participants/36 generated excerpts; relevant external evidence, not validation on this corpus. Original collection-dependent fitting needs a frozen train-only adaptation. |
| MIDI-VAE L0093; MuseMorphose L0098 | Separate style classifiers/latent conditioning; MuseMorphose controls rhythm intensity/polyphony and reports chroma/groove fidelity. Current structure checks cover part of that vocabulary. Neither guarantees exact melody nor equates attributes with composer identity. |
| L0027/L0103/L0067 | Sequence/context and target/counterexample approaches differ from centroids; L0103 already uses music21 features. Gaussian67 borrows L0067 satisfaction, not its full system or composer validation. Sequence objectives remain deferred. |
| L0150/L0185/L0207-L0209 | Cycle loss is soft fidelity; global style plus temporal melody supports separation; METEOR supports melody/texture control. Pretrained composer generation/steering differs from protected transfer and has asset/ASAP-overlap issues. Retain E4.6 NO-GO; no new neural architecture justified now. |

### Existing tools: inspected interfaces and source, not executed

| Integration | Useful role | Feasibility and limits |
|---|---|---|
| [jSymbolic](https://jmir.sourceforge.net/jSymbolic.html), [CLI](https://jmir.sourceforge.net/manuals/jSymbolic_manual/commandline_files/commandline.html), [installation](https://jmir.sourceforge.net/manuals/jSymbolic_manual/installation_files/installation.html) | External MIDI baseline; pitch/rhythm/chord/texture descriptors; train-only classifier evaluator using existing fitting machinery. Pinned JAR plus libraries/config; ACE XML definitions/values and CSV. | Java 8+ required; local 1.8.0_291 available, JAR absent. Moderate adapter cost; runtime unmeasured. Exclude IDs/constant piano instrumentation/performance confounds. Not a pretrained evaluator or transformation engine; modern features need not match Herremans' old subset. |
| [StyleRank API](https://github.com/jeffreyjohnens/style_rank/blob/master/src/style_rank/api.py), [bindings](https://github.com/jeffreyjohnens/style_rank/blob/master/src/style_rank/bindings.cpp), [build](https://github.com/jeffreyjohnens/style_rank/blob/master/setup.py) | Reuse categorical extraction and RF leaf similarity. Source inspected at upstream tip `649266e`. | C++/pybind11, Windows unverified. API learns vocabulary/forests jointly from rank/style sets; defaults 100 trees/depth3 differ from paper 500/depth5, seed unspecified. Bindings omit <=10-chord inputs. Freeze training vocabulary/forests, retain failures and normalize reference sizes. Moderate-high adaptation cost. |
| [music21 chordification](https://music21.org/music21docs/usersGuide/usersGuide_09_chordify.html), [key analysis](https://music21.org/music21docs/moduleReference/moduleAnalysisDiscrete.html), [jSymbolic port](https://music21.org/music21docs/moduleReference/moduleFeaturesJSymbolic.html) | Harmony/tonality/voice/texture diagnostics and score transformation utilities. Already in isolated musif environment. | MIDI spelling/voice/key inference needs review; XML is richer but a separate study. Partial feature port contains unimplemented entries, not equivalent to Java jSymbolic. Re-exported score edits cannot automatically preserve raw MIDI contracts. |
| Existing musif; [Groove2Groove evaluation modules](https://github.com/cifkao/groove2groove/tree/master/code/groove2groove/eval) | Retain proven adapter; consider XML separately. Reuse `notes_chroma_similarity.py` for harmonic content, inspect profiles for parity. | musif numeric projection omits much stock harmony/texture and needs a training schema. Upstream metrics import note_seq/pretty_midi/museflow/confugue and use old NumPy APIs; isolate metric reuse rather than installing legacy TensorFlow model stack. |

## 5. Two bounded tracks and ONE transfer candidate

### A. Minimal external validation

Use the pinned jSymbolic CLI/export adapter and existing classifier-fitting machinery to compare external descriptors with custom93 on the same frozen inner-training/validation works. First check schema/coverage/repeatability on the predeclared nine training samples; then fit the same classifier configuration on each representation and score real inner-validation works plus the 18 existing V2-05 outputs. Fit preprocessing only on training, keep failed inputs/matched denominators, exclude IDs and constant piano instrumentation. Freeze the external evaluator before any real B search; never optimize it. Report whether attribution and movement conclusions change with representation, including uncertainty/disagreement. Neither representation is ground truth.

**Bound:** one external backend/configuration and one existing classifier; no full E1c, combined feature search, StyleRank, new metric or outer-test access. Do not load full-corpus caches or frozen E2/E3 held-out outputs. A small development comparison can reveal dependence but cannot establish corpus-wide equivalence. Cost: low-moderate, roughly 1-2 focused adapter/verification passes plus measured extraction/fitting; no GPU.

### B. Compare transformation mechanisms, select one

| Literature approach | What could change the transfer itself | Repository fit / cost / selection |
|---|---|---|
| Herremans L0034: jSymbolic classifier-guided optimization | Optimize composer affinity with musically constrained pitch neighborhoods, rather than only classify outputs. | Reuse fitting/constraints; swapping features alone leaves E3's edit space unchanged. Adding pitch neighborhoods plus repeated jSymbolic extraction costs more and makes A's evaluator objective-dependent. Plausible, but not the selected implementation. |
| Groove2Groove L0074: content/style separation and accompaniment transfer | Separate protected content from a style-conditioned accompaniment decoder, replacing E3's marginal-histogram edits with generated accompaniment patterns. | Borrow the separation principle. Upstream model uses chord-chart content/BIAB styles, not protected piano melody or composer labels; legacy Python3.6/TensorFlow1.12 and unverified compatible weights raise cost. A pattern-remapping shortcut would be a new heuristic, not a reproduction. High cost; defer the model. [Upstream](https://github.com/cifkao/groove2groove). |
| **Zalkow/Brand/Graf L0103: local modification search (selected)** | Individual-note pitch replacement, bounded local neighborhoods and visited-state memory can change accompaniment intervals/chord voicing while keeping source events fixed. | Adapt the paper's pitch neighborhood, not its monophonic bass/chord-conditioned Markov system. Reuse existing scoring/content machinery; moderate cost, roughly 3-4 focused implementation/verification passes plus pilot runtime. No new neural model or sequence metric. |

Basis: stored L0034/L0074 and [L0103.pdf](../../Literatura/selected/L0103.pdf), especially its local moves and optimization sections. These are three compared approaches, but only L0103 is proposed as a new transfer candidate.

**Concrete adaptation.** E3's `transformation.py` edits timing/duration/thinning/octave doubling; with zero transposition it cannot replace individual accompaniment pitches. Add an external local-search adapter with a stable `NoteId -> pitch` edit map, replacing the five-gene GA/proposal mechanism for this candidate without changing frozen E3. Propose one unprotected note's pitch within +/-4 semitones of its ORIGINAL pitch (the paper's major-third neighborhood); keep every onset, duration, velocity, channel and note count fixed. Predeclare a cumulative edit ceiling, initially 20% of accompaniment notes. Reject out-of-range pitches, newly introduced ambiguous same-channel/pitch overlaps and notes overtopping protected attacks. Keep an immutable original melody mask; no global transposition or split/join/rhythm edits in this first adaptation.

Reuse `e3/structure.py` bars/IDs/mask, `validate_constraints`, MIDI serialization, V2-03 raw-event checks, fit/leakage guards and V2-05 cache/history/accounting concepts. Accept feasible moves that improve the already implemented train-only logistic93 target affinity; retain visited states and identity fallback. Freeze that objective before either search. This is a deliberately reduced L0103 adaptation, not its full tabu/Pareto/Markov reproduction or guaranteed musical grammar. jSymbolic stays reserved for evaluation. Existing harmony diagnostics/listening must expose harmful pitch edits rather than silently assuming constraints imply musical quality.

**Decision experiment, proposed only:** select two lexical validation work groups per composer, one sample/work, before inspecting scores: six sources, 12 directed tasks. Compare identity, current zero-transposition E3 and the single new candidate. Both searches use the same frozen logistic93 objective, fits/content contract and seeds 1729/1730: 48 searches. Cap each at 512 proposals (E3 population32/generations16, no early stop), including identity; separately report unique evaluations, cache hits, rejections and runtime. This compares complete mechanisms, not a pure optimizer-only ablation.

Continue only if exact observable content passes, the candidate yields usable non-identity outputs, target movement improves over E3 under the reserved external evaluator across multiple source works/directions, and blinded paired listening finds no accompanying loss of melody recognition/musical quality. Agreeing with time_pitch or increasing the optimized classifier alone is insufficient. Preregister the actual continuation thresholds/rater design before execution; report this small pilot descriptively, not as confirmatory proof. No usable external signal means A is inconclusive; it does not make time_pitch a substitute ground truth or justify more search.

**Recommendation:** implement the small external check AND investigate accompaniment pitch search as the next package of work. A's integration/protocol work and B's synthetic operator/constraint work can proceed independently; real B search waits for the frozen validation protocol. This adds a testable transformation contribution instead of another evaluator-only milestone. Retain V2 provenance, content checks, baseline results and serialization lessons; defer additional objectives, StyleRank and neural candidates. Neither track is implemented or authorized to run in this revision.

## Execution boundary and checkpoint

At the initial consolidation audit, the entry tree was clean, HEAD `6895b0c`, tracking `origin/refactor/research-v2`; origin uniquely identified `paniedziela/musicians-style-modeling`. Branch changes contained source/tests/docs/lock, no generated scientific artifacts; targeted secret-pattern scan found no matches. Normal explicit push to the corresponding branch returned **Everything up-to-date**. No force, merge, rebase, tag, branch switch or main/master change. This document is left uncommitted for review.

Audit inputs: tracked code/docs/history, local literature and public upstream source. No dataset/cache/fitted-model/per-output scientific files were opened; no scientific audits/tests/optimizations rerun, outer-test data accessed, libraries installed, V2-06 added or proposed solution implemented. Only this decision document was added. Historical records govern what happened; this document governs the next-step recommendation.
