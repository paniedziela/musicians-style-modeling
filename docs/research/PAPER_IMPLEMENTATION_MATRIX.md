# PAPER_IMPLEMENTATION_MATRIX — all 13 selected sources

Current checkpoint (2026-10-05): A+B implementation reviewed and logically committed; [review/checkpoint report](AB_IMPLEMENTATION_REPORT.md). The user authorized a clean-tree branch push followed by Track A only under [its reviewed protocol](AB_A_PROTOCOL.md). jSymbolic feature/schema/model settings remain frozen. No B execution, outer-test source/features, main change, merge or rebase. Stop for review after A; historical entries below retain their original boundaries.

Updated 2026-10-04. All selected PDFs were read in the preceding audit; this revision retains substantive findings, records their byte identities and completes the master prompt's 19 per-paper fields. PDF versions take precedence over stale bibliography labels. Availability means a repository/link was located in that audit, not that dependencies, weights or an end-to-end reproduction were verified. No paper implementation begins in V2-01. All locations below are future capability-oriented locations; none is a parallel Research V2 package.

Priority is usefulness for this thesis. Estimates describe bounded adaptations using agent passes and measured/expected compute, not elapsed calendar time. Exact schedules/costs require a concrete task and environment. Literature results use their original protocols and are not directly comparable to E1 grouped-work scores.

## L0027 — Automatic Modeling of Musical Style — Lartillot, Dubnov, Assayag, Bejerano (2001)

Source/version: `Literatura/selected/L0027.pdf`; 8 PDF pages; method and musical examples. SHA256: `f3d00a326db03cecfef9933667968ebb4ebf8aafde431f008560310b6091876c`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | Unsupervised style learning and generation from symbolic sequences. |
| Representation | MIDI onset/offset slices and encoded pitch/rhythm/velocity patterns. |
| Definition of style | Repeated variable-length sequential regularities, rather than one global feature centroid. |
| Definition of content | No explicit source-melody preservation contract; continuation/generation from learned material. |
| Features | Symbolic tokens, repeated subsequences and context statistics. |
| Model/search algorithm | Incremental Parsing / Lempel–Ziv-related modelling and prediction suffix trees. |
| Objective/loss | Corpus-conditioned sequence prediction; no differentiable transfer loss. |
| Datasets | Illustrative Bach, Chopin, Satie and jazz material; no comparable grouped benchmark. |
| Evaluation | Musical examples and analysis; no frozen composer attribution or transfer test. |
| Subjective evaluation | Illustrations/author discussion; no formal panel or participant count established. |
| Available code/checkpoints | Historical OpenMusic-related implementation discussed; current source/checkpoints not verified. |
| Assumptions | Tokenization retains relevant sequential structure; enough examples per context. |
| Limitations for this thesis | Polyphony encoding and score-MIDI timing differ from monophonic pattern modelling; sparse corpus. |
| Reproducible component | Sequence/context scoring after specifying a reproducible token schema. |
| Adaptable component | Train-only pattern/Markov score as a later contrasting style definition. |
| Expected effort | MEDIUM; 2–3 agent passes for bounded scorer; CPU tokenization/context fitting, cost benchmark required; may parallelize after schema freeze. |
| Dependencies | V2-01 and event/token contract; optional after E1d, not a V2-01 dependency. |
| Priority / usage | COULD / ADAPT_METHOD; full historical environment DEFER. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/sequence.py`; no new code now. |

## L0034 — Classification and Generation of Composer-Specific Music Using Global Feature Models and Variable Neighborhood Search — Herremans, Sörensen, Martens (2015)

Source/version: `Literatura/selected/L0034.pdf`; 21 PDF pages; classifier and FuX/VNS experiments. SHA256: `3de5876317dcfc5216b6178e0f2eec5e75add32ba272da73a504fe42f90c1de4`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | Composer classification and classifier-guided pitch generation. |
| Representation | Symbolic scores; pitch edits over rhythm held fixed. |
| Definition of style | Composer-discriminative global feature model. |
| Definition of content | Counterpoint rules and fixed rhythmic skeleton; not exact source melody protection. |
| Features | 111 jSymbolic descriptors, 12 selected pitch/interval features. |
| Model/search algorithm | Logistic regression / SVM classifiers and variable neighborhood search in FuX. |
| Objective/loss | Counterpoint costs plus weighted composer-probability penalty; classifier-only generation can be musically poor. |
| Datasets | 1,045 KernScores: 595 Bach, 254 Haydn, 196 Beethoven. |
| Evaluation | Stratified 10-fold results: LR about 83% accuracy / 94% AUC; SVM 86% / 96%; not our grouped-work protocol. |
| Subjective evaluation | Author analysis/listenable 16-bar examples; no formal listening panel established. |
| Available code/checkpoints | Paper describes FuX availability; matching source/version/checkpoints not verified here. |
| Assumptions | Global descriptors generalize from corpus to edited candidates. |
| Limitations for this thesis | Classifier exploitation, fixed rhythm and corpus dependence; generation differs from protected-melody transfer. |
| Reproducible component | Train-only classifier affinity; preserve our grouped folds rather than reproducing stratified scores. |
| Adaptable component | Probability/log-odds objective with a separate evaluator and explicit dependence reporting. |
| Expected effort | MEDIUM; 1–2 passes for scorer; CPU fold fitting, no transfer search until V2-05; parallel scoring after fit protocol freeze. |
| Dependencies | V2-01/V2-03 and frozen custom93/E1 protocol; jSymbolic only for a separate feature comparison. |
| Priority / usage | SHOULD / BORROW_OBJECTIVE; retain as methodological predecessor. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/classifier.py`; wrap E1 fitting, no copy. |

## L0067 — Evolutionary music composition system with statistically modeled criteria — Kowalczuk, Tatara, Bąk (2017)

Source/version: `Literatura/selected/L0067.pdf`; 12 PDF pages; GA criteria and online evaluation. SHA256: `56fa4d911da1a2f20986f3a5560e9c0c924720996842ad034d13c46a574c9ea9`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | Evolve monophonic melodies under rules/statistically modelled criteria. |
| Representation | 12 bars, 4/4, sixteenth-note grid, fixed chord scheme. |
| Definition of style | Desired feature distributions/weights, not demonstrated composer attribution. |
| Definition of content | Fixed length/chord framework; no existing protected source melody. |
| Features | 11 normalized melodic/statistical criteria plus rules. |
| Model/search algorithm | GA with six mutation operators; population 512, 500 epochs. |
| Objective/loss | Weighted Gaussian criterion satisfaction plus rule fitness; reference or manually tuned means/variances. |
| Datasets | 12 reference melodies of mixed provenance plus generated examples. |
| Evaluation | Fitness behavior and rating comparisons, not leakage-safe composer transfer. |
| Subjective evaluation | 39 online raters, nine melodies; pleasantness and VAD ratings; reported pleasantness roughly 2.15–3.28. |
| Available code/checkpoints | Matching implementation/checkpoints not verified. |
| Assumptions | Feature marginals and weights meaningfully characterize desired music. |
| Limitations for this thesis | Marginal independence/weight tuning, monophony/grid mismatch; Gaussian density is not automatically style validity. |
| Reproducible component | Gaussian feature-profile score with explicit variance floor. |
| Adaptable component | Train-only corpus-derived profiles, family balancing, degeneracy checks in E1d. |
| Expected effort | LOW; 1–2 passes; cached feature fitting/scoring on CPU, no GA now; can run alongside other E1d scorers. |
| Dependencies | V2-01/V2-03; fixed feature schema and training folds. |
| Priority / usage | SHOULD / BORROW_OBJECTIVE; full GA reproduction DEFER. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/gaussian.py`. |

## L0074 — Groove2Groove: One-Shot Music Style Transfer with Supervision from Synthetic Data — Cífka, Şimşekli, Richard (2020)

Source/version: `Literatura/selected/L0074.pdf`; 14 PDF pages including cover; event-profile/content evaluation. SHA256: `e9daf4cd0a73f5886cca4301a28344610d6395e7f2364264946ee0c92bd175c3`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | One-shot accompaniment style transfer with synthetic paired supervision. |
| Representation | Symbolic accompaniment; chord content and style encoders with CNN/GRU/attention. |
| Definition of style | Accompaniment event-pattern profiles and conditioning example. |
| Definition of content | Harmonic/chord content; not exact protected melody. |
| Features | Time–pitch profile: 24×41, time 0–4 beats / interval −20..20; onset–duration: 24×12, onset mod 4 / duration 0–2 beats; velocity/drum profiles; frame chroma. |
| Model/search algorithm | Groove2Groove conditional sequence model trained on Band-in-a-Box synthetic styles. |
| Objective/loss | Paired sequence cross-entropy; evaluation separates style cosine profiles and harmonic chroma. |
| Datasets | 1,476 BIAB styles (20 each reserved for validation/test); 1,200 training chord charts × 252 bars; 600 validation and 600 test charts × 16 bars; Bodhidharma 660 files / 8,934 segments. |
| Evaluation | Macro/nano aggregation of event-profile cosine; 12 chroma frames/beat with two-beat smoothing; quantitative synthetic and real-example tests. |
| Subjective evaluation | Author/example inspection; no formal listener panel established. |
| Available code/checkpoints | [Author repository](https://github.com/cifkao/groove2groove) located; full local model/checkpoint reproduction not performed. |
| Assumptions | 4/4 or 12/8 regular accompaniment and recognizable harmony; synthesis style labels. |
| Limitations for this thesis | Composer piano scores include other meters, longer notes and sparse/irregular events; instrument/velocity often uninformative. |
| Reproducible component | Explicit event-profile definitions and cosine measurements. |
| Adaptable component | Piano/composer profiles with predeclared meter/window/overflow/empty-profile policy, separate from exact melody checks. |
| Expected effort | MEDIUM; 2–3 passes for metric; CPU event parsing/caching, no neural training; profiles can cache independently after schema freeze. |
| Dependencies | V2-01/V2-03 and score-MIDI event contract; not dependent on synthetic training assets for metric adaptation. |
| Priority / usage | MUST / BORROW_METRIC; full model DEFER. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/event_profiles.py`; harmonic measurements in `content_metrics/harmony.py` when needed. |

## L0093 — MIDI-VAE: Modeling Dynamics and Instrumentation of Music with Applications to Style Transfer — Brunner et al. (2018)

Source/version: `Literatura/selected/L0093.pdf`; 8 PDF pages; architecture and latent-style examples. SHA256: `3f98224d79a009d3a4a9fcc6d4d76da9d4537bac7cc945692f2b80211420e6af`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | Style-conditioned MIDI reconstruction, latent interpolation and transfer. |
| Representation | Four monophonic voices, fixed bars / 16 steps, 60 pitches, velocity and instrument channels. |
| Definition of style | Latent style classification dimensions / domain labels. |
| Definition of content | Shared encoded musical material; no exact note-event protection guarantee. |
| Features | Pitch, velocity, instrumentation; separate diagnostic classifiers. |
| Model/search algorithm | GRU VAE, latent size 256; parallel encoders/decoders and style coordinate substitution. |
| Objective/loss | Reconstruction CE/MSE, style CE and KL (beta 0.1). |
| Datasets | 1,989 files: 477 classical, 554 jazz, 659 pop, 156 Bach, 143 Mozart; 90/10 song split. |
| Evaluation | Reconstruction/style classifier diagnostics and examples; constrained short excerpts. |
| Subjective evaluation | Informal examples; no formal listener panel established. |
| Available code/checkpoints | [Author repository](https://github.com/brunnergino/MIDI-VAE) located; old Keras/Theano stack; compatible weights not verified. |
| Assumptions | Fixed voices, quantization, short context and meaningful dynamics/instrument differences. |
| Limitations for this thesis | Lossy score representation and instrumentation domain mismatch; exact pitches not preserved. |
| Reproducible component | Historical architecture only; full reproduction would need old environment/data contract. |
| Adaptable component | Latent-strength controls as a possible future baseline, not core infrastructure. |
| Expected effort | HIGH for model; at least 4 passes; paper reports roughly 48 h on GTX1080 for 400 epochs per pair; our cost unmeasured; evaluation design can proceed separately. |
| Dependencies | Neural GO review, compatible representation/assets and stronger content/style evaluation. |
| Priority / usage | DEFER / RELATED_WORK_ONLY; possible BASELINE after feasibility review. |
| Exact proposed repository location | Conditional future `src/musicians_style/neural_baselines/midi_vae.py`, never during V2-01. |

## L0098 — MuseMorphose: Full-Song and Fine-Grained Piano Music Style Transfer with One Transformer VAE — Wu & Yang

Source/version: `Literatura/selected/L0098.pdf`; local arXiv:2105.04090v3, December 2022; 16 pages; TASLP 2023. SHA256: `661d09bb776974cf75cab35777f18b062a4f3e2e1df43be56884f533f9567c92`. The local PDF is the audited source; online code availability does not imply a runnable verified reproduction.

| Required field | Audited finding / decision |
|---|---|
| Task | Full-song bar-level controllable piano transfer. |
| Representation | REMI with bar/chord/tempo/velocity; bar latent codes and long-context Transformer decoder. |
| Definition of style | Named rhythmic-intensity and polyphony controls in eight bins. |
| Definition of content | Encoded musical structure; chroma/groove fidelity, not exact original melody events. |
| Features | Rhythm intensity, polyphony, chroma and groove profiles. |
| Model/search algorithm | 79.4M-parameter Transformer VAE; bar encoding and in-attention conditioning. |
| Objective/loss | Reconstruction, KL with free bits and cyclic scheduling; attribute-conditioned decoding. |
| Datasets | AILabs-Pop1K7: 1,747 songs / 108 h; LPD subset: 10,626 songs / 650 h; specified validation/test splits. |
| Evaluation | Attribute Spearman fidelity, content chroma/groove, perplexity and diversity. |
| Subjective evaluation | Examples/discussion; no formal panel established in the audited paper. |
| Available code/checkpoints | [Author repository](https://github.com/YatingMusic/MuseMorphose), linked [Zenodo assets](https://zenodo.org/records/5119525); weights not locally downloaded or validated. |
| Assumptions | Quantized bars and attributes estimated from sufficient piano data. |
| Limitations for this thesis | Attributes are not composer identity; pop/corpus and representation mismatch; neural fidelity is soft. |
| Reproducible component | Named controllable attribute definitions. |
| Adaptable component | Rhythm/polyphony measurements and per-attribute trade-off reporting. |
| Expected effort | MEDIUM metric adaptation, 1–2 passes / CPU parsing; HIGH model adaptation, 4+ passes / GPU cost unmeasured; metrics may parallelize after contract freeze. |
| Dependencies | V2-03 for metrics; separate neural GO/assets for model. |
| Priority / usage | SHOULD / BORROW_METRIC; full model DEFER. |
| Exact proposed repository location | Future `src/musicians_style/content_metrics/attributes.py`; optional `neural_baselines/musemorphose.py`. |

## L0103 — Musical Style Modification as an Optimization Problem — Zalkow, Brand, Graf (2016)

Source/version: `Literatura/selected/L0103.pdf`; 6 PDF pages as duplicated spreads; unique spreads 1/3/5 contain printed pp. 1–6. SHA256: `c46d8093faa3bc89895593046d3f71da0b649ba491080d736232214d122e069a`.

| Required field | Audited finding / decision |
|---|---|
| Task | Locally modify an existing bass line toward a target and away from counterexample style. |
| Representation | Monophonic pitch/duration with fixed chord annotations. |
| Definition of style | Target/counterexample global classifier and sequential Markov regularities. |
| Definition of content | Local edits and fixed harmonic framework; no exact primary-melody guarantee. |
| Features | 410 dimensions: 324 music21 plus 86 custom; Markov orders 0–4. |
| Model/search algorithm | Tabu search with local pitch ±major-third edits, split/join and multiobjective acceptance. |
| Objective/loss | Classifier score and interpolated additively smoothed target/counterexample Markov ratio; accepted moves nonworsen all objectives. A log-ratio subtraction would be an adaptation, not literal reproduction. |
| Datasets | Jaco: eight pieces / 2,227.5 quarter-note duration; Victor: eight / 3,642.75. |
| Evaluation | Amazing Grace/New Britain case study; 3,398 trials, 14 accepted transformations. |
| Subjective evaluation | Author analysis, no formal listening panel established. |
| Available code/checkpoints | Paper/thesis implementation discussed; current matching public code/checkpoints not verified. |
| Assumptions | Chord labels and enough sequential reference data; restrictive Pareto moves remain useful. |
| Limitations for this thesis | Monophonic bass versus polyphonic piano; small corpus and possibly sparse counts. |
| Reproducible component | Target/counterexample scoring and traceable local/Pareto improvement. |
| Adaptable component | Leakage-safe relative classifier/sequence scoring around existing E3, without duplicating the optimizer. |
| Expected effort | MEDIUM; 2–3 passes per bounded scorer; CPU context fitting/cached scores, search cost deferred; may parallelize after token/fold contract. |
| Dependencies | V2-01/V2-03 and E1d; tokenization for sequence score, train-only counterexample definition. |
| Priority / usage | SHOULD / BORROW_OBJECTIVE, ADAPT_METHOD. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/relative.py` or `sequence.py`; external pilot runner later. |

## L0150 — Symbolic Music Genre Transfer with CycleGAN — Brunner, Wang, Wattenhofer, Zhao (2018)

Source/version: `Literatura/selected/L0150.pdf`; 8 PDF pages; CycleGAN and classifier experiments. SHA256: `2aa78444a0d580959da85785eb1d9eb3cd614200ab09cfa369d859f338d7f935`.

| Required field | Audited finding / decision |
|---|---|
| Task | Unpaired jazz/classical/pop symbolic genre transfer. |
| Representation | Binary 64×84 piano rolls, four 4/4 bars, merged tracks and uniform export velocity 127. |
| Definition of style | Genre-domain distribution learned by discriminators. |
| Definition of content | Cycle reconstruction/fidelity, not exact source-note preservation. |
| Features | Piano-roll occupancy and diagnostic CNN classifiers. |
| Model/search algorithm | CycleGAN plus additional mixed-domain discriminators. |
| Objective/loss | Least-squares GAN and L1 cycle loss; extra discriminator trade-off. |
| Datasets | 12,341 jazz / 16,545 classical / 20,780 pop phrases; 90/10 split; grouped source-work separation not established. |
| Evaluation | Classifier/noise robustness and style/content trade-off on phrases. |
| Subjective evaluation | Informal examples; no formal listening panel established. |
| Available code/checkpoints | [Author repository](https://github.com/sumuzhao/CycleGAN-Music-Style-Transfer) located, TensorFlow 1.x; matching weights not verified. |
| Assumptions | Fixed-meter short quantized phrases adequately represent domains. |
| Limitations for this thesis | Genre is not composer; lossy quantization/track merging; exact melody absent. |
| Reproducible component | Historical neural loss/baseline only; short-roll reconstruction is insufficient for our content contract. |
| Adaptable component | Content/style trade-off and cycle-loss caution in related work. |
| Expected effort | HIGH for faithful model; 4+ passes and GPU training/data reconstruction cost unmeasured; design audit can parallelize, no training authorized. |
| Dependencies | Separate neural GO, compatible assets/representation and evaluation. |
| Priority / usage | DEFER / RELATED_WORK_ONLY; possible BASELINE only after review. |
| Exact proposed repository location | Conditional future `src/musicians_style/neural_baselines/cyclegan.py`. |

## L0185 — Encoding Musical Style with Transformer Autoencoders — Choi et al. (2020)

Source/version: `Literatura/selected/L0185.pdf`; local arXiv:1912.05537; 13 pages; ICML 2020. SHA256: `bd096223165f42a8de2cb57e6f6534738c8f93a8ed89131768af7969c9d047ad`.

| Required field | Audited finding / decision |
|---|---|
| Task | Encode performance style and combine with temporal melody for generation/transfer. |
| Representation | Performance event stream with 10-ms time and velocity; 92-token melody at 100-ms resolution. |
| Definition of style | Global learned style code, with musical feature distribution diagnostics. |
| Definition of content | Temporally distributed Viterbi melody, not note-on Skyline or exact source events. |
| Features | Eight distributions: density, range, pitch mean/std, velocity mean/std and duration mean/std, evaluated in two-second windows. |
| Model/search algorithm | Deterministic Transformer autoencoder, mean pooled style and melody conditioning. |
| Objective/loss | Maximum likelihood with noise/augmentation including pitch ±6 and tempo ±5%. |
| Datasets | MAESTRO v1 roughly 1,100 performances / 80:10:10; private YouTube corpus about 400k performances / 10k h. |
| Evaluation | Gaussian overlap area for content/style feature distributions, reported separately. |
| Subjective evaluation | 492 style judgments and 714 melody/style ratings (judgments, not participant counts); melody-only vs combined comparison p=0.0894. |
| Available code/checkpoints | [Legacy Magenta project](https://github.com/tensorflow/magenta) is paper-linked; exact runnable code revision/checkpoint availability unresolved. |
| Assumptions | Performance dynamics are meaningful; temporally compressed melody retains content. |
| Limitations for this thesis | Score MIDI dynamics and tempo differ from expressive performance; style pooling cannot guarantee exact pitches. |
| Reproducible component | Distribution-overlap diagnostic definitions, with degenerate variance handling. |
| Adaptable component | Content/style axes and windowed distribution comparisons; do not average the two axes. |
| Expected effort | MEDIUM metrics, 2–3 passes / CPU windows; HIGH model, 4+ passes / GPU and private-data constraints; metric work may parallelize after schema freeze. |
| Dependencies | V2-03 for metric adaptation; separate neural/data access review for model. |
| Priority / usage | SHOULD / BORROW_METRIC, ADAPT_METHOD; full model DEFER. |
| Exact proposed repository location | Future `src/musicians_style/content_metrics/distributions.py`; performance dimensions require a separate experiment. |

## L0207 — From Generality to Mastery: Composer-Style Symbolic Music Generation via Large-Scale Pre-training — Yao and Chen (2025)

Source/version: `Literatura/selected/L0207.pdf`; local arXiv:2506.17497; 15 pages; GnM. SHA256: `6dcd7864d4e53e7ba907e3c49ef96e93a51c1c084340c02b236e723fd0090a3e`.

| Required field | Audited finding / decision |
|---|---|
| Task | Composer-conditioned generation from scratch through broad pretraining and composer adaptation. |
| Representation | REMI beat subdivision 12; meter handling includes remapping unsupported cases. |
| Definition of style | Composer-conditioned adapted model with five-layer adapter. |
| Definition of content | Primer/structural controls; no protected source-piece transfer contract. |
| Features | Classifier affinity, groove, entropy, structure, chord ranking and embedding distance. |
| Model/search algorithm | 12-layer, 46M-parameter Transformer, hidden size 512. |
| Objective/loss | Autoregressive token loss for pretraining/fine-tuning. |
| Datasets | Nine pretraining datasets; about 1.28M bars / 64.8M tokens excluding four target composers; fine-tuning 891 works: Bach 307, Mozart 161, Beethoven 236, Chopin 187. |
| Evaluation | Composer classifier 77.6%; structural/entropy/groove and Fréchet embedding diagnostics. |
| Subjective evaluation | 54 raters, ten two-alternative composer questions and 1–5 musicality; pretraining improves quality/specificity trade-off. |
| Available code/checkpoints | [Author repository](https://github.com/AndyWeasley2004/Generality-to-Mastery) located with linked weights/datasets; no local download or validation. |
| Assumptions | Broad pretraining helps scarce composer data; conversion/meter policy is acceptable. |
| Limitations for this thesis | Generation rather than controlled transfer; possible data overlap and different composer/domain coverage. |
| Reproducible component | Pretrained baseline only after verifying training overlap and assets. |
| Adaptable component | Composer strength/adapter rationale; generation diagnostics as secondary evidence. |
| Expected effort | HIGH; 4+ passes for model integration; paper RTX3090 / 120k pretraining and 28k adaptation steps; our runtime unmeasured; overlap audit can run independently. |
| Dependencies | Neural GO, weights/license, corpus-overlap audit and shared evaluation. |
| Priority / usage | DEFER / RELATED_WORK_ONLY; optional BASELINE. |
| Exact proposed repository location | Conditional future `src/musicians_style/neural_baselines/gnm.py`. |

## L0208 — METEOR: Melody-aware Texture-controllable Symbolic Music Re-Orchestration via Transformer VAE — Le & Yang (2025)

Source/version: `Literatura/selected/L0208.pdf`; local arXiv:2409.11753v3; 9 pages; IJCAI 2025. SHA256: `4004e1e6886669bfc4f7d5025dd35afc0061289821393bf8e07cbfc458707a0e`.

| Required field | Audited finding / decision |
|---|---|
| Task | Melody-aware texture-controlled re-orchestration and lead-sheet generation. |
| Representation | REMI+ with pitch class/octave and bar/track headers; instruments and texture bins. |
| Definition of style | Per-track density/rhythmicity/polyphony/mean-pitch controls, not composer identity. |
| Definition of content | Melody track chosen by highest mean pitch; forced melody tokens but ornaments can be added. |
| Features | Texture attributes, token melody similarity, chroma, Spearman and Jensen–Shannon diagnostics. |
| Model/search algorithm | 67M Transformer VAE with melody guiding. |
| Objective/loss | Conditional reconstruction/latent objective; guide melody during decoding. |
| Datasets | SymphonyNet around 46k pieces; paper training about a week on RTX6000 24GB. |
| Evaluation | 20 output objective comparisons; best-track melody similarity about 0.632 guided vs 0.491 unguided. |
| Subjective evaluation | 24 raters on six re-orchestrations; 13 raters on four lead-sheet examples; random four-condition groups and 0–5 ratings. |
| Available code/checkpoints | [Author repository](https://github.com/dinhviettoanle/meteor) located, linked Hugging Face assets; weights not locally validated. |
| Assumptions | Track-level melody and texture controls fit orchestral data. |
| Limitations for this thesis | Highest-mean-pitch track differs from event Skyline; added ornaments and token edit distance do not establish exact protected melody. |
| Reproducible component | Melody/content similarity and separate texture-control diagnostics. |
| Adaptable component | Explicit melody-guiding/content evaluation principle; exact event protection must remain a harder thesis contract. |
| Expected effort | MEDIUM metrics, 2–3 passes / CPU parsing; HIGH model, 4+ passes / GPU cost unmeasured; metric work can parallelize after content contract. |
| Dependencies | V2-03 for metrics; compatible orchestral representation/assets and neural review for baseline. |
| Priority / usage | SHOULD / BORROW_METRIC, ADAPT_METHOD; model DEFER. |
| Exact proposed repository location | Future `src/musicians_style/content_metrics/melody.py`; conditional `neural_baselines/meteor.py`. |

## L0209 — Composer Vector: Style-steering Symbolic Music Generation in a Latent Space — Jiang et al.

Source/version: `Literatura/selected/L0209.pdf`; local arXiv:2604.03333v1 dated 2026-04-03; 15 pages; NeurIPS 2025 workshop header. SHA256: `63cb7bd093920d9806e17314f1473492e17b292909374c13ecad3252f29bf60a`.

| Required field | Audited finding / decision |
|---|---|
| Task | Inference-time latent composer steering, strength control and style fusion/suppression. |
| Representation | ABC music through pretrained NotaGen/ChatMusician models. |
| Definition of style | Mean last-token hidden-state composer vectors at layers chosen through probes/clustering. |
| Definition of content | Autoregressive format/content continuity; no exact protected source melody contract. |
| Features | Hidden states; external composer classifier; CLaMP3 symbolic and CLAP audio similarities. |
| Model/search algorithm | Add scaled composer vector at selected layer (e.g. 19/29), restore activation norm, avoid format tokens. |
| Objective/loss | No steering training loss; alpha-scaled activation intervention. |
| Datasets | ASAP XML→ABC, 11 target composers. Local paper has inconsistent external-corpus counts (236/15 versus 222/16 in related totals); retain uncertainty. |
| Evaluation | Linear probe 75/25 stratified split; external midi-classical classifier 70/10/20, CLaMP3 about 89.38%; work-group separation not established. |
| Subjective evaluation | Qualitative examples; no formal listening panel established. |
| Available code/checkpoints | [Author repository](https://github.com/JiangXunyi/Composer-Vector) located with minimal README; base checkpoints separate; exact compatible assets not verified. Online author/year metadata differs from local PDF; cite local version explicitly. |
| Assumptions | Composer directions encode usable style independently of formatting. |
| Limitations for this thesis | ASAP overlap with thesis corpus is critical; generation/steering rather than exact transfer; bibliographic/count discrepancies. |
| Reproducible component | Probe/vector extraction only with accessible base model and overlap audit. |
| Adaptable component | Continuous style strength and format-token exclusions as later design ideas. |
| Expected effort | HIGH; 4+ passes for model integration; base-model inference/embedding extraction costly, benchmark required; provenance/overlap checks can parallelize. |
| Dependencies | Neural GO, exact PDF/model revision, weights/license, ASAP overlap and held-out evaluator. |
| Priority / usage | DEFER / RELATED_WORK_ONLY; optional BASELINE after feasibility review. |
| Exact proposed repository location | Conditional future `src/musicians_style/neural_baselines/composer_vector.py`. |

## L0230 — Quantifying Musical Style: Ranking Symbolic Music Based on Similarity to a Style — Ens & Pasquier (2019)

Source/version: `Literatura/selected/L0230.pdf`; 8 PDF pages; ISMIR 2019 / arXiv posting 2020. SHA256: `c99b2389ccef46d6058b7f09b8f1cdeff3d653ad9bf250e02b515413751720dd`.

| Required field | Audited finding / decision |
|---|---|
| Task | Corpus-relative symbolic style ranking (StyleRank). |
| Representation | Categorical pitch/interval/chord-set/voice-leading and melodic n-gram features. |
| Definition of style | Random-forest leaf co-occurrence similarity to a reference style corpus. |
| Definition of content | No source-content protection; ranking only. |
| Features | 352 pitch-class sets including ties; voice leading/melody patterns; sparse vocabulary capped at 1,000 where specified. |
| Model/search algorithm | Per-feature RF: 500 trees, depth 5, entropy, balanced classes; one-hot leaves and cosine similarity averaged over target references. |
| Objective/loss | Discriminate target corpus C from ranked collection G, then reference-leaf similarity; applying a frozen train-only target-vs-nontarget model is an adaptation. |
| Datasets | Classical Archives: 75 composers / six genres; dedup using first/last-100-event Levenshtein threshold 0.75; reference sizes 10–100 across 1,000 trials. |
| Evaluation | Ranking/classification robustness; BachBot 36 outputs / 630 pairwise comparisons related to human Bach misidentification. |
| Subjective evaluation | BachBot survey 5,967 participants; ranking compared to pre-existing human judgments, not a direct new style-transfer panel. |
| Available code/checkpoints | [Author repository](https://github.com/jeffreyjohnens/style_rank) located, C++/pybind11; environment/reproduction not locally verified. |
| Assumptions | Reference/collection discrimination and categorical similarity are musically useful. |
| Limitations for this thesis | Pitch-centric descriptors; representation/corpus dependence remains; original collection-fitting is unsuitable for blindly scoring held-out transformed outputs. |
| Reproducible component | Leaf similarity construction with fixed vocabulary and leakage checks. |
| Adaptable component | Fit target/nontarget training works only and freeze forests/vocabulary before output scoring; report adaptation explicitly. |
| Expected effort | MEDIUM; 2–3 passes for bounded scorer; CPU feature-forest fitting may be costly, benchmark/cache required; per-fold jobs may parallelize after fit protocol. |
| Dependencies | V2-01/V2-03; concrete categorical schema; optional E1d extension, compiled dependencies only if wrapping upstream. |
| Priority / usage | SHOULD / BORROW_METRIC; BORROW_OBJECTIVE only after validity review. |
| Exact proposed repository location | Future `src/musicians_style/style_metrics/stylerank.py`. |

## Cross-paper synthesis and decision

Interpretable/optimization family (L0027/L0034/L0067/L0103/L0230): corpora define sequence, classifier, distribution or relative-ranking scores; each is a competing operational definition. Favor bounded score/metric adaptation, train-only fitting and traceable local search before reproducing whole systems. Gaussian marginals and classifiers can both be exploited; use held-out validity rather than name recognition.

Neural style/content family (L0074/L0093/L0098/L0150/L0185): separation requires explicit representations and two evaluation axes. Cycle/reconstruction losses are soft fidelity signals. Named rhythm/texture controls and event-profile measurements recur across independently motivated papers and are immediately more useful than new neural architectures. Preserve exact note events separately from harmonic/attribute similarity.

Recent composer/control family (L0207/L0208/L0209): pretraining, melody guiding and continuous steering offer useful ideas, but generation, orchestration and corpus overlap differ from protected composer transfer. METEOR's melody principle fits the thesis more closely than unconstrained generation; it still does not guarantee exact protected notes. Accessible code does not establish compatible weights/data or bounded compute.

E1c: compare frozen custom93, musif, jSymbolic, selected music21 and schema-defined combinations under frozen grouped E1 folds where feasible. V2-02 is a nine-sample feasibility gate only. XML coverage supports a separate matched score-aware E1m study with musif/Partitura. Do not modify custom93 or synthesize a new split for failed extractors.

E1d: begin with concrete train-only RMS/Gaussian/classifier and event-profile candidates, retain relative/StyleRank/sequence as bounded later extensions. RF is a separate held-out evaluator with shared-corpus/feature dependence; event profiles provide a more structurally distinct perspective. Do not describe every candidate as independent or combine scores by arbitrary averaging. V2-05 objective selection is provisional until predeclared validity criteria pass; a failing metric need not be optimized.

Only V2-01 is implemented now. No feature-backend/content-metric/style-metric/neural modules above are created in this pass. The five-task backlog, files, acceptance gates, estimates and stop boundaries are in ROADMAP.md.
