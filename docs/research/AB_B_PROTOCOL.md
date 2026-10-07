# Track B pilot protocol - 2026-10-06

**Authorized by the current user request for ONE unchanged B pilot.**
Before execution, simplify only the listening/continuation boundary. Gates 1-3
and the frozen scientific experiment remain unchanged. Report B and stop; no
automatic follow-up, listening execution, commit or push is authorized here.

## Preflight

Read committed RESEARCH_V2_DIRECTION_DECISION, AB_IMPLEMENTATION_CONTRACT,
AB_IMPLEMENTATION_REPORT, AB_A_COMPLETION and relevant V2-05 protocol/completion.
Historical draft preflight recorded a clean tree; this task entered with only
the untracked B draft. HEAD and local origin tracking ref both
`61e23542ca22f0b714e1d2ec3910f881a35b6c22`, branch `refactor/research-v2`.

Independently verified current metadata fingerprints, lexical prospective cohort,
both fits' train/forbidden identities, backend distribution/config/schema pins
and Java 1.8.0_291. No original corpus MIDI, outer-test features or full feature
cache read. Existing focused synthetic tests: **27 passed**, 13 existing warnings,
14.16 s, including full workflow/dispatch and 200-example property verification.
Command: `.venv/Scripts/python.exe -B -m pytest -q -p no:cacheprovider -o addopts=
tests/unit/test_accompaniment_search.py tests/unit/test_research_ab.py
tests/property/accompaniment/test_pitch_invariants.py --hypothesis-profile=full`.

Verified SHA256 fingerprints:

- A `experiments/research_ab_a_evaluation_reviewed/frozen_external.joblib`:
  `d25f17836e32ab2c963a7bfec34f0e84f7c5723831e6d1ee8c1d5a337e01d9a3`.
- A fit metadata: `05c33e82a057c28d0eb2acc0acd1045612ff29d22888faa665fcda27995ecbcb`.
- Objective `experiments/research_v2_05_pilot_2026-10-05/fits/inner00.joblib`:
  `665171f9285fe4721b9060daf7a23bb9fd6082d72b6e667e98b95b972a5feec4`.
- Manifest: `84b93d2a57430a3a9697f174110acf428534116fcad33a9166a65ad4e8b6bda4`.
- Splits: `e92f7791541398971b4410e542aeb37c66831b885d1e2e39b495e08dc5e7458a`.

At entry the existing B review had approval=false and null continuation
thresholds, listening design and external-bundle hash. This protocol supplies
gates 1-3 before any B results. The current user request authorizes execution
and supersedes earlier requirements to freeze a listening design before B.
The review retains listening_design=null; the runner no longer requires it.
The existing synthetic workflow checks this boundary with the same B dispatch.

## Unchanged experiment

Frozen r00/f00/inner00, TRAIN76/45 works and validation44/24 works. Cohort matches
previous metadata-only `future_cohort_plan.json`: Bach fugues BWV846/BWV848;
Beethoven sonata13 movement4/sonata16 movement1; Chopin Ballade2/Berceuse op57.
Exact sample IDs: `bach--fugue-bwv-846`, `bach--fugue-bwv-848`,
`beethoven--piano-sonatas-13-4`, `beethoven--piano-sonatas-16-1`,
`chopin--ballades-2`, `chopin--berceuse-op-57`. These are the first two lexical
validation work groups per composer and first lexical sample per group.

Each source targets the other two composers: 12 directed tasks. Compare original
identity, existing V2-05-adapted zero-transposition E3, and implemented
L0103-inspired accompaniment pitch search, seeds1729/1730: **48 searches**.
Each uses exactly **512 proposals including identity**. E3 population32,
generations16, elite2/tournament3, stagnation17, inherited mutation draws/settings,
545 evaluation requests; pitch search512 requests, full visited memory and strict
improvement >1e-12. Report cache/rejection/unique-evaluation differences separately.

Both optimize the SAME frozen V2-05 DEVELOPMENT logistic93 target probability:
TRAIN-only VarianceThreshold(0), StandardScaler, C1/balanced/lbfgs/L2,
max_iter5000/tol1e-4/model-seed1729. No fitting in B. jSymbolic's exact frozen A
model (457 TRAIN-retained dimensions) is RESERVED evaluation-only; no external
scores may guide proposals, selection, stopping or retries.

Common hard policy: original Skyline protected absolute pitches/onsets/offs and
velocity; protected duration may be ambiguous but never failed/undefined;
essential metadata/structural checks and frozen E3 feasibility. Same-tick order
and Skyline reselection remain diagnostic. Identity preserves original bytes.

Candidate additionally fixes ALL note onsets/durations/velocities/channels/count,
track/event positions/order, delta times, controllers and metadata. Only paired
raw on/off pitch numbers change, using immutable ORIGINAL NoteIds/mask, ORIGINAL
pitch +/-4, MIDI range and cumulative floor(0.20 * accompaniment_count) ceiling.
Reject new same-channel/pitch overlaps including cross-track, overtopping protected
attacks and raw/reparsed lineage differences. Unmatched/unmapped sources fall back
to identity. E3 retains its existing accompaniment edit space/serialization; its
comparator is governed by the common protected-content policy, rather than the
candidate's additional all-note/raw invariants.

## Frozen continuation gates and pilot disposition

**The numerical gates below are research-policy decisions, not literature-derived
facts or validated musical thresholds.** None is selected from B results. The
existing literature inspiration does not validate these continuation choices.

1. Technical gate: 48/48 searches complete within the frozen budgets with selected
   outputs passing common protected-content/structural requirements; 24/24 candidate
   outputs pass additional raw/all-note invariants. Verified content violations
   mean NO-GO. Execution/accounting failures leave B INCONCLUSIVE, without rerunning.
2. Usability gate: candidate non-identity under BOTH seeds on at least 9/12 tasks
   (at least 18/24 outputs). All identity fallbacks stay in every denominator.
   With valid complete execution, failure of this gate means NO-GO for this mechanism/budget.
3. Reserved signal gate: all six identity and 48 output external extractions/scores
   available and finite. Average the two seeds per task. At least 8/12 tasks must
   have BOTH candidate movement from identity and paired candidate-minus-E3 movement
   >1e-12, spanning at least four source works, four of six composer directions
   and all three source composers. The mean of BOTH movements across all12 tasks
   must also be positive separately for EACH seed. Ties/saturation do not win;
   missing values never disappear or count as wins. No best-seed choice.

Disposition after this single pilot: gates 1-3 passing means PROMISING, with
musical benefit still unestablished. Gates 1-2 passing but gate 3 failing or
unavailable means INCONCLUSIVE. A verified content violation or complete valid
execution failing gate 2 means NO-GO; execution/accounting failures remain
INCONCLUSIVE. Stop after reporting in every case. None of these outcomes
approves listening, another experiment or broader development.

Use all12 task rows with both seeds and six-work descriptive summaries; seeds are
not independent works. No confirmatory p-values or confidence-based promotion.
jSymbolic is not perceptual ground truth; disclose A's substantial TRAIN-range
extrapolation, uncalibrated/saturated probabilities and shared corpus. No clipping,
calibration or output-selected validity thresholds. Logistic gains alone cannot
pass. time_pitch/onset_duration remain secondary diagnostics, never ground truth.

## Provisional future listening work

A blinded paired listening comparison may be considered in a separately approved
future task. It is outside this pilot and will not be implemented or executed
now. Rater count, eligibility, excerpts, rendering, assignment and decision
criteria remain provisional and must be reviewed if that future study is
requested. This stage is not bound to eight raters or the former listening
pass/NO-GO cutoffs. No perceptual success or absence of musical harm can be
claimed from the present technical/classifier results.

## Execution/reporting boundary

Under the current user authorization, bind this document's SHA256 and gates 1-3
into the local B review, preserve backend/input/both-fit fingerprints, retain
listening_design=null and record separate future approval, then set approval=true.
Persist protocol before optimization. Use the existing implementation-report
`pilot-b` command and its fresh destination `experiments/research_ab_b_pilot_reviewed`.
If destination exists, stop and audit; never pick a second path to rerun the pilot.

Report: content/invariant and ambiguity rates; non-identity/fallback coverage;
rejection reasons (E3 reasons may overlap); proposal/request/unique/cache accounting;
optimized logistic93; reserved identity/E3/candidate movement by source/direction/seed;
frozen V2-05 time_pitch/onset_duration on selected outputs only; runtime and source/
fit preservation audit. Use existing scoring functions/fits, no new automated metric.
Keep missing/failing outputs and zero identity gains. Compare complete mechanisms,
not a pure optimizer ablation. Better optimized logistic93 alone is insufficient.

No outer-test original/features, full-corpus cache, real reruns, tuning of radius,
ceiling/objective/evaluator/cohort/budget/seeds/gates, added search method
or V2 stage. Stop after B with PROMISING / INCONCLUSIVE / NO-GO. Keep generated
scientific artifacts local. No commit/push, main merge or listening execution
is part of this request. The only runner change removes the listening-design
requirement from the review boundary; search and evaluation are unchanged.
