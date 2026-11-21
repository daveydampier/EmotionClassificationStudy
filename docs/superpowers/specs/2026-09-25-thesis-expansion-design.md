# Thesis Expansion Design: The Cost of Going Lightweight, Revisited

**Date:** 2026-09-25
**Status:** Approved. Amended 2026-11-21 after implementation planning (amendment log in §5). Implementation is organized by the roadmap in `docs/superpowers/plans/2026-11-21-thesis-expansion-epic.md`, which also carries the current schedule.
**Scope:** Two official semesters (Spring 2027, Summer 2027) plus an unofficial Fall 2026 for engineering. Compute: H200 GPU server plus local CPU.

This document turns the finished independent study (results in `results/paper_final/`)
into a thesis-scale research program. It is the backbone for the Fall 2026 proposal and
the reference for every engineering change in this repo. Chapter numbers below refer to
the thesis's four *results* chapters, not to the whole manuscript.

---

## 0. Starting point and why it is not enough

The independent study compared seven models across three tiers on GoEmotions (native-27,
Ekman-6, sentiment-3), with three seeds, bootstrap CIs, a training-budget check for the
transformers, and a cross-dataset transfer test against SemEval-2018. Its findings:

1. TF-IDF + LogReg sits within ~0.045 macro-F1 of BERT-base at roughly 1/30 the size
   and 1/60 the training time.
2. The lightweight penalty is concentrated in rare emotions, where CIs are also widest.
3. Cross-corpus transfer is corpus- and direction-specific, not a function of training
   size or model class; the linear model transfers more stably than the transformers.
4. The advisor asked for a prescriptive model-selection framework (Implications section).

A thesis committee will push on five soft spots. Each is closed by this design.

| Soft spot | Where it is closed |
|---|---|
| Fixed 0.5 decision threshold, default hyperparameters, validation split never used | §2.3 protocol, Chapter 1 |
| "Corpus-specific transfer" rests on two corpora, one direction pair | §2.1 corpora, Chapter 4 |
| The deep-learning middle tier is a single BiLSTM | §2.2 models, Chapter 2 |
| Three seeds, no paired significance test between models | §2.3 protocol |
| Cost measured in seconds on mixed hardware; no energy | §2.3 protocol |

### Positioning against prior work

Two prior papers sit closest to this work and must be cited and differentiated.

- **arXiv:2601.18162** (January 2026) compares TF-IDF LogReg, BiLSTM, and BERT on
  GoEmotions with per-emotion scores. It measures no efficiency axis and sweeps no label
  granularity. Chapters 1 and 2 differ on exactly those points.
- **Bostan and Klinger, COLING 2018**, *An Analysis of Annotated Corpora for Emotion
  Classification in Text*, already ran a train-on-one, test-on-another matrix over 14
  emotion corpora with a MaxEnt bag-of-words classifier. A cross-corpus matrix is
  therefore not a contribution in itself. Chapter 4's contribution is what that study
  could not ask with one model: how transfer differs across model classes and routes,
  whether measurable domain distance predicts it, and whether a selection rubric derived
  from it holds on an unseen corpus.

### Alternatives considered

- **B. Depth-first (three routes only, transfer at three corpora).** Rejected: leaves
  the transfer claim under-tested and gives the advisor's framework ask only a
  descriptive treatment.
- **C. Framework-first (transfer matrix + validated rubric, thin model set).** Rejected
  as the primary plan but kept as the fallback shape: if time runs short, this design
  degrades to C by cutting Chapter 3, not Chapter 4.
- **A. Full program (chosen).** Four results chapters, staged so it degrades to C.

---

## 1. Research questions

**Thesis question.** When a deployment cannot afford a full transformer, which route to
a lightweight emotion classifier should it take, what does that choice cost per emotion,
and can the right choice be predicted for a corpus the framework has never seen?

Each chapter answers one question and is built around one headline figure.

### Chapter 1: Rigor-fixed replication
**RQ1.** Do the study's findings survive validation-tuned thresholds and
hyperparameters, energy as the cost axis, and paired significance tests?

- Headline figure: per-emotion LogReg-vs-BERT F1 delta with paired bootstrap CIs,
  significant emotions marked after Holm correction.
- Supporting: the study's Pareto plots re-issued on an energy axis; a before/after
  threshold-tuning figure (fixed 0.5 vs tuned, per model, macro-F1 and rare-emotion F1).
- Success: every table in the independent study is reproduced under the new protocol,
  with the study's published numbers and the new-protocol fixed-0.5 numbers retained
  alongside as the "before" columns (§2.3, item 9).

### Chapter 2: Three routes to lightweight
**RQ2.** At matched size, latency, and energy, which route wins: small by design, frozen
encoder plus linear head, or compressed transformer? Where do the frontiers cross?

- Headline figure: three Pareto frontiers (one per route) on size and on energy axes,
  with the reference-ceiling and hosted-tier points plotted for context.
- Supporting: iso-budget table at three size tiers (§2.4); per-emotion heatmap by route.
- Success: frontier crossings, if any, are reported with CIs; each tier names a winner.

### Chapter 3: Mechanism of the rare-emotion penalty (flex chapter)
**RQ3.** How much of the rare-emotion penalty is label scarcity, label ambiguity, or
decision policy?

Four independent sub-analyses, ordered cheapest first. Any subset still reads as a chapter.

- 3a Human ceiling: model F1 divided by human-rater F1, per emotion.
- 3b Calibration: reliability diagrams and ECE before/after temperature scaling.
- 3c Noise robustness: F1 degradation curves under six perturbation types.
- 3d Learning curves: F1 vs training fraction per route, split rare vs frequent emotions.
- Success: each completed sub-analysis yields one figure and one paragraph of mechanism.

### Chapter 4: Transfer matrix and validated selection framework (protected chapter)
**RQ4.** Is cross-corpus transfer predicted by corpus distance and source breadth, and
does a rubric derived on five corpora pick the right model on the sixth?

- Prior art: Bostan and Klinger (2018) ran a cross-corpus matrix with a single MaxEnt
  classifier (§0). The matrix here is the instrument, not the finding. The findings are
  the per-route differences, the distance regression, and the validated rubric.
- Headline figure: regret bar chart, leave-one-corpus-out, rubric vs two naive rules.
- Supporting: 6x6 transfer heatmaps per route; transfer F1 vs domain distance scatter;
  the framework rendered as a decision tree.
- Success: mean regret reported with CIs against always-BERT and always-LogReg.

Chapter 4 is protected because it answers the advisor's ask. Chapter 3 is the flex.

---

## 2. Experimental design

### 2.1 Corpora

Six core corpora and one optional. All are harmonized to Ekman-6 for cross-corpus work
through `labels.py`. In-domain results stay at each corpus's native granularity.

| Corpus | Genre | Native labels | Splits | Notes |
|---|---|---|---|---|
| GoEmotions | Reddit comments | 27 + neutral, multi-label | official train/val/test | Primary. The `raw` config carries per-rater rows for the human ceiling (3a). |
| SemEval-2018 Task 1 E-c | Tweets | 11, multi-label | official | Already wired (community Parquet mirror; provenance noted). |
| dair-ai/emotion | Tweets | 6, single-label | official | Advisor's list. |
| XED (English) | Movie subtitles (OPUS OpenSubtitles) | 8 (Plutchik), multi-label | none official; 70/10/20 at seed 42, stratified on each line's rarest label | Öhman et al., COLING 2020; CC BY 4.0; about 17,500 annotated lines. Anticipation and trust have no Ekman equivalent and are dropped, as for SemEval-2018. The Hugging Face repo is script-based, so the tab-separated file is read from the Helsinki-NLP/XED GitHub repository. The separately shipped neutral lines are not used. |
| ISEAR | First-person situation narratives | 7, single-label | none official; stratified 70/10/20 at seed 42 | shame and guilt map to sadness, consistent with remorse and embarrassment in the existing GoEmotions Ekman map. |
| MELD, text only | TV dialogue utterances | 7 (Ekman-6 + neutral) | official | Advisor's list. Text field only; audio/video out of scope. |
| EmpatheticDialogues (optional) | Crowd-sourced dialogue | 32, single-label | official | Second fine-grained corpus; enables a 27-vs-32 fine-grained transfer probe. Dropped first if it fails to load. |

Rules:
- **Shared-emotion rule.** Cross-corpus macro-F1 is computed over the Ekman emotions
  present in *both* the source and target corpus, neutral excluded. This generalizes the
  study's neutral-exclusion rule. Every transfer cell reports its shared-emotion count.
- **Ekman coverage.** GoEmotions, SemEval-2018, XED, and MELD cover all six Ekman
  emotions. dair-ai/emotion has no disgust and ISEAR has no surprise, so the smallest
  shared set is four emotions (dair-ai/emotion paired with ISEAR).
- **Independence.** Core corpora must not share source texts. TweetEval-emotion was in
  the first draft and was removed for this reason: it is derived from SemEval-2018
  Task 1, so the two would put overlapping tweets on both sides of a transfer cell.
- **Single-label corpora** are treated as multi-hot with one active label so the existing
  binary-relevance pipeline applies unchanged. Subset accuracy then equals accuracy.
- **Provenance.** Any corpus loaded from a community mirror gets a provenance note in the
  results README, as SemEval-2018 has today.
- **Fallback.** If a corpus cannot be loaded or its license is unclear, drop to five
  corpora. Chapter 4's leave-one-corpus-out validation is valid at five.

### 2.2 Models, grouped by route

About twenty configurations. Tables carry all of them. Figures show one representative
per route (marked *) to stay readable.

**Route S: small by design**
- Naive Bayes (ComplementNB), LogReg*, LinearSVM, RandomForest, XGBoost, each on TF-IDF.
- GloVe-average embeddings + LogReg.
- TextCNN (Kim 2014) and BiLSTM (existing).
- ELECTRA-small and MiniLM-L6 fine-tuned end-to-end (via the existing `hf_<name>` wrapper).

**Route F: frozen encoder + linear head**
- all-MiniLM-L6-v2 embeddings + LogReg*.
- bge-small-en embeddings + LogReg.

**Route C: compressed transformer**
- DistilBERT-base (distillation arm; existing)*.
- BERT-base with int8 dynamic quantization (post-training, no retraining).
- BERT-base magnitude-pruned to 50% of weights, then fine-tuned for one epoch to recover.

**Reference ceiling**
- BERT-base* (existing), RoBERTa-base, ModernBERT-base.

**Hosted, no-training tier** (Chapters 2 and 3b only; evaluated on test sets, never trained)
- One open LLM on the H200: Qwen2.5-7B-Instruct, zero-shot and 5-shot.
- One API LLM: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`), zero-shot and 5-shot.
- One hosted decision model: **Jev** (TypeSafe AI), pinned to `jev-1.13.0` or the stable
  version at evaluation time, with the version and evaluation date recorded. Jev is not
  an LLM: it takes a block of text plus typed questions and returns typed answers with a
  probability and a confidence score, all questions evaluated in parallel and in
  isolation. Multi-label emotion is posed as one Noul ("is this statement true")
  question per emotion in a single call, which is binary relevance, the same decision
  rule every other model uses. Single-label corpora may additionally use the Choice
  primitive. Zero-shot only; no fine-tuning or few-shot support exists. Jev's claims of
  calibrated probabilities, low cost, and low latency map directly onto the scorecard's
  ECE, cost, and latency axes, and it has no published text-classification evaluation,
  so it enters as an untested data point, not as a route to lightweight.
- Recorded for every hosted model: macro/micro-F1, per-emotion F1, ECE, end-to-end
  latency (network round-trip included and stated as such), and cost per 1,000
  predictions (API list price, or H200 energy for the open model). Prompts, question
  templates, and few-shot examples are fixed and committed. If a full test set would
  exceed a per-corpus dollar cap of USD 25, evaluate on a stratified 2,000-example
  subsample and say so.
- Hosted models have no local model size. They are plotted on the energy, latency, and
  cost axes and appear in the iso-budget tables under "unlimited" size.

### 2.3 Protocol (applies to every run)

1. **Validation split.** Every corpus has one (official where it exists, see §2.1). No
   model or threshold is ever tuned on the test split.
2. **Per-label threshold tuning.** Decision thresholds are tuned per label on validation
   to maximize per-label F1, then frozen for test. Chapter 1 reports fixed-0.5 and tuned
   side by side; every other chapter reports tuned only.
3. **Equal hyperparameter budget.** Eight configurations per model, random search at
   seed 42. The selection metric is validation macro-F1 computed at per-label thresholds
   tuned on that same validation split. The procedure is identical for every model, so
   models with poorly scaled probabilities are not penalized by a fixed 0.5. The selected
   configuration is then retrained across the protocol's seeds and its thresholds are
   re-tuned on validation. The test split is never read during search. Models with fewer
   than eight meaningful configurations use their full grid. Search spaces:
   - LogReg / LinearSVM / frozen-encoder heads: C in log-space [0.01, 100].
   - NB: alpha in [0.01, 10]. RF: n_estimators {100, 200, 400} x max_depth {None, 20, 50}.
     XGBoost: depth {4, 6, 8} x learning_rate {0.05, 0.1, 0.3} x n_estimators {200, 400}.
   - BiLSTM / TextCNN: lr {1e-3, 3e-4} x hidden {128, 256} x dropout {0.2, 0.5}.
   - Transformers: lr {1e-5, 2e-5, 3e-5, 5e-5} x epochs {3, 5}. Batch size fixed at 32.
4. **Seeds.** Five seeds (42 to 46) for every model with stochastic training (BiLSTM,
   TextCNN, RF, XGBoost, all transformers). Deterministic models run once. Bootstrap CIs
   (1,000 resamples) on every test-set metric, as now.
5. **Paired significance.** For every model-vs-model comparison on a shared test set: a
   paired bootstrap of the F1 difference, per emotion and for macro-F1, with Holm
   correction across emotions. Reported wherever a delta is claimed.
6. **Energy.** CodeCarbon wraps training and a fixed 1,000-example inference pass. GPU
   readings come from NVML and are treated as measurements. CPU readings on Windows are
   thermal-design-power estimates and are labeled as estimates in every table. The
   `device` column stays on every scorecard row.
7. **Calibration.** Temperature scaling fit on validation for every model; ECE reported
   before and after. Chapter 3b analyzes it; every scorecard carries both columns.
8. **Hardware statement.** Training time and latency are reported on the hardware each
   route would realistically use (classical on CPU, neural on GPU), stated per row.
   Energy and model size are the cross-hardware comparison axes.
9. **Transformer fine-tuning recipe.** AdamW, batch size 32, maximum length 128, linear
   warmup over the first 10% of steps followed by linear decay to zero. This replaces
   the independent study's loop (constant learning rate, batch 16, no schedule), so the
   ceiling numbers can move for reasons unrelated to thresholds. Chapter 1 therefore
   reports three columns per model: the study's published number, the new protocol at a
   fixed 0.5, and the new protocol with tuned thresholds.

### 2.4 Chapter-specific runs

**Chapter 1.** The study's seven models plus the rest of routes S and F, on GoEmotions
native-27, Ekman-6, and sentiment-3 under the §2.3 protocol. This subsumes the study's
granularity sweep and feature comparison. The route C compression arms and the extra
ceilings (RoBERTa, ModernBERT) are built in March (§3) and folded into the Chapter 1
tables then; the chapter's claims rest on the seven-model core, which is built and run
first.

**Chapter 2: iso-budget tiers.** Three size tiers anchored on named models so the rule is
unambiguous: Tier 1 = every model no larger than the frozen MiniLM-L6 pipeline (the
smallest neural encoder deployed); Tier 2 = every model no larger than DistilBERT-base
(the compressed transformer); Tier 3 = every model no larger than BERT-base (the full
reference). Everything larger (RandomForest, RoBERTa, ModernBERT) is "unlimited". Model
size is the on-disk size of the serialized model with sparse tensors stored sparsely, so
pruning counts. Within each tier the winner per route is the configuration with the
best validation macro-F1, reported on test. The same three tiers are cut on inference
energy per 1,000 examples using the same anchors, and both cuts are reported.

**Chapter 3a: human ceiling.** From the GoEmotions `raw` config, restricted to examples in
the official test split with at least three raters: each rater's label set is scored
against the consensus of the other raters, where an emotion is in the consensus if at
least half of the other raters marked it. Aggregating over all (example, rater)
pairs gives per-emotion rater precision, recall, and F1, plus Krippendorff's alpha per
emotion. The normalized model score is model F1 divided by rater F1 for that emotion.

**Chapter 3b: calibration.** Reliability diagrams and ECE before/after temperature
scaling for one representative per route, on GoEmotions native-27, plus Jev, whose
calibration claim is tested here. Temperature scaling is applied to Jev's returned
probabilities exactly as to any other model's.

**Chapter 3c: noise robustness.** Six perturbation types applied to the GoEmotions test
set: character typos, adjacent-key swaps, casing changes, emoji insertion, slang and
abbreviation substitution, random word deletion. Three intensities: 10%, 20%, 30% of
tokens affected. The perturbed sets are registered as evaluation sets, so each trained
model predicts on them in the same process that trained it (§2.5). Nothing is retrained
for this analysis; transformer checkpoints may be saved so that perturbed sets built
later can still be scored. Perturbation seeds are fixed and the perturbed sets are
cached under `data/processed/`.

**Chapter 3d: learning curves.** GoEmotions native-27 at seven training fractions (1, 2,
5, 10, 25, 50, 100%), three seeds (reduced from the protocol's five to bound the sweep),
one representative per route plus BERT-base. Curves
reported for macro-F1 and separately for the rare-emotion subset (support below 100 in
the test split).

**Chapter 4: transfer matrix.** Every model of §2.2 (excluding the hosted tier) is trained on
each corpus in Ekman-6 and evaluated on every other corpus under the shared-emotion rule,
giving a 6x6 (or 7x7) matrix per model. Domain distance between every corpus pair is
measured four ways: Jensen-Shannon divergence between unigram distributions over the
union of the top 20,000 tokens; cosine distance between mean all-MiniLM-L6-v2 sentence
embeddings; log ratio of mean token length; Jensen-Shannon divergence between Ekman label
distributions. Transfer F1 is regressed on these four distances plus log source-corpus
size, with (model, source, target) as the unit of analysis.

**Chapter 4: selection framework.**
- *Constraint profile.* Five dimensions: size tier (1, 2, 3, unlimited); inference-energy
  tier (1, 2, 3, unlimited); metric priority (macro-F1 or micro-F1); calibration
  requirement (post-scaling ECE at most 0.05, or none); expected domain shift (in-domain
  or shifted). That is 4 x 4 x 2 x 2 x 2 = 128 profiles.
- *Derivation.* On the derivation corpora, for each profile the winner on a corpus is the
  feasible model with the best priority metric (in-domain score for "in-domain" profiles;
  mean transfer score into that corpus for "shifted" profiles). The rubric's
  recommendation for a profile is the model that wins on the most derivation corpora,
  ties broken by mean normalized scorecard score.
- *Validation.* Leave-one-corpus-out over the six corpora. Regret on the held-out corpus
  for a profile is the priority metric of the actual best feasible model minus that of
  the recommended model. Report mean regret over profiles and folds, with bootstrap CIs,
  against two naive rules: always-BERT and always-LogReg (each clipped to feasibility).
- *Presentation.* The derived rubric is rendered as a decision tree keyed on the five
  profile dimensions. This is the artifact the advisor asked for.
- *Hosted models.* Excluded from the framework's first version, because their
  deployability differs in kind: a network dependency, and text leaving the premises.
  If the hosted tier wins on cost or calibration in Chapter 2, a sixth boolean
  dimension, "hosted allowed", is added, doubling the profile count to 256.

### 2.5 Repository changes

Extend the existing package; do not rewrite it. Metrics, the scorecard, results
persistence, and the model interface carry over. One structural change is required,
because today the runner trains, scores at a fixed threshold, and discards the
per-example probabilities.

- **Fit once, predict on many, cache.** A trained model predicts on every named
  evaluation set in the same process: its corpus's validation and test splits, every
  other corpus's test split, and every perturbed test set. The probabilities are written
  to a prediction cache with a manifest. Threshold tuning, calibration, significance
  tests, the transfer matrix, and the framework are then analyses over the cache, with
  no retraining.
- **No general model persistence.** Evaluation sets must exist before training; a
  missing one costs a retrain. Transformers get an optional saved checkpoint.
  Compression arms fine-tune their base model inside `fit()`, so their training cost
  includes the base.
- **Predictions are artifacts.** The cache lives under `results/thesis/predictions/`,
  git-ignored, and is archived with a DOI for publication. Scorecards, per-emotion CSVs,
  and figures stay committed.

| Area | Change |
|---|---|
| `corpora.py` (new) | Corpus registry: loader, native labels, Ekman map, single- or multi-label flag, split policy, stem prefix. Schema resolution moves here from the CLI, and the shared-emotion rule is applied per corpus pair. |
| `models/registry.py` (new) | Model registry: builder, route, search space, stochastic flag. Replaces the if-chain in `scripts/run_experiment.py`. |
| `predictions.py` (new) | Save and load cached probabilities with their manifest (model, parameters, seed, train corpus, schema, evaluation-set name, timings, energy, device, size). |
| `loaders.py` | Five new loaders (dair-ai, XED, ISEAR, MELD text, EmpatheticDialogues) plus a GoEmotions raw-rater loader. Each normalizes to the existing `EmotionDataset` form. |
| `labels.py` | Per-corpus native label lists and native-to-Ekman-6 maps; a shared-emotion intersection helper; tests for every map. |
| `experiment.py` | Split into `fit_and_predict()` (train once, predict on every named evaluation set, capture timings, energy, device, size) and `score()` (scorecard rows from cached predictions, with threshold mode and calibration applied). `run_experiment()` stays as a wrapper. Also the eight-configuration search, CodeCarbon energy capture, and the new scorecard columns (`train_energy_wh`, `infer_energy_wh_per_1k`, `ece_calibrated`, `threshold_mode`, `energy_measured` flag). |
| `metrics.py` and sibling modules | Pure functions over `(y_true, y_prob)` arrays: per-label threshold tuning; temperature scaling; paired bootstrap delta with Holm correction; shared-label macro-F1; rater-agreement metrics (per-emotion rater F1, Krippendorff's alpha). |
| `models/transformer.py` | The §2.3 fine-tuning recipe; optional checkpoint saving. |
| `models/embeddings.py` (new) | Frozen sentence-encoder + LogReg, and GloVe-average + LogReg. |
| `models/compressed.py` (new) | Int8 dynamic quantization and magnitude-pruning wrappers around the HuggingFace model. |
| `models/hosted.py` (new) | Prompted classifier for the open and API LLMs, and a Jev client that poses one Noul question per emotion per call; fixed templates, few-shot examples, version pinning, and cost accounting. |
| `models/deep.py`, `models/classical.py` | Add TextCNN; add XGBoost. |
| `scripts/` | `run_transfer_matrix.py`, `run_learning_curves.py`, `run_noise.py`, `make_perturbed_sets.py`, `make_framework.py` (derive + leave-one-corpus-out validate + render tree). |
| `viz.py` | Route frontiers; transfer heatmap; F1-vs-human-F1 scatter; learning curves; degradation curves; reliability diagrams; regret chart; decision-tree render. |
| `results/thesis/` (new) | Parallel to `paper_final/`, same stem convention, with new dataset prefixes `dair_`, `xed_`, `isear_`, `meld_`, `ed_` and a file key README. The `predictions/` subfolder is git-ignored. |
| `tests/` | One test module per new loader, metric, and model; CI stays network-free. |
| `pyproject.toml` | New extras: `codecarbon`, `sentence-transformers`, `xgboost`, `nlpaug`, `krippendorff`, `optimum` or `torch.ao` for quantization, LLM and TypeSafe client libraries. |

Run scripts for the GPU server follow the existing `run_m*.sh` pattern, one per chapter.

### 2.6 Out of scope

Multilingual work (BRIGHTER / SemEval-2025), the multimodal parts of MELD, speech,
fine-tuning or distilling from LLM labels, and peak-RAM measurement. Each is a
possible follow-on and none is needed to answer the four research questions.

---

## 3. Schedule

> **Superseded 2026-11-21.** None of the fall work below had started by that date. The
> current schedule is section 6 of the roadmap. The original plan is kept for reference.

**Fall 2026, unofficial (October to December): engineering and Chapter 1.**
- October: five loaders and the rater loader; validation split, threshold tuning,
  hyperparameter search, energy capture, paired tests; literature refresh; join the Jev
  early-access waitlist.
- November: the embeddings and hosted model modules, TextCNN, XGBoost, small transformers;
  Chapter 1 re-run on the H200 (seven-model core plus routes S and F); proposal draft
  built from this document.
- December: proposal to the committee; Chapter 1 results written; buffer.

**Spring 2027, official: the two chapters that carry the thesis.**
- January to February: in-domain runs on all corpora, all models, five seeds; cross-corpus
  evaluation; domain-distance features.
- March: the compression module and route C arms, RoBERTa and ModernBERT ceilings, run
  on all corpora and folded into Chapters 1 and 4; iso-budget analysis; hosted tier.
- April to May: Chapters 2 and 4 results written; framework derived.

**Summer 2027, official: mechanism, validation, manuscript.**
- June: 3a human ceiling, 3b calibration, 3c noise (cheap, first); 3d learning curves last.
- July: leave-one-corpus-out validation; all figures final; full draft.
- August: revisions and defense preparation.

Compute is not the bottleneck: BERT-base took about four minutes for five epochs on
GoEmotions on the H200, so the full in-domain matrix is on the order of one GPU day.
Analysis and writing are the bottleneck, which is why each chapter is written in the
month its results land.

---

## 4. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A corpus fails to load or has unclear licensing (XED's Hugging Face repo is script-based; EmpatheticDialogues and MELD are script- or GitHub-hosted). | Read XED from the file in its GitHub repository. Otherwise use a mirror with a provenance note, as for SemEval-2018. Drop to five corpora; Chapter 4 is valid at five. EmpatheticDialogues is dropped first. |
| Threshold tuning changes the study's headline. | Reported as a finding. Chapter 1 keeps fixed-0.5 and tuned side by side. |
| Ekman-6 harmonization loses signal (love to joy, guilt to sadness). | In-domain results stay native; every transfer cell states its shared-emotion count. |
| CPU energy is an estimate on Windows. | Labeled as an estimate everywhere; iso-budget tiers lean on model size and GPU energy. |
| Hosted-tier evaluation cost. | Per-corpus dollar cap with a stratified subsample fallback (§2.2). Jev's list price makes a full GoEmotions test pass well under a dollar. |
| Jev early access never arrives, or the hosted model changes under evaluation. | Waitlist joined in October, tier not needed until March. If access fails, the tier keeps the two LLMs and Chapter 2 loses one row. Version and evaluation date are recorded on every hosted-tier result. |
| Scope creep. | Chapter 3 flexes; 3d learning curves are cut first, then 3c. Chapter 4 is never cut. |
| Writing time squeezed by experiments. | Each chapter is written in the month its results land; the proposal is drafted from this document in November. |

---

## 5. Amendment log

**2026-11-21, after implementation planning.** Reasons and evidence are in section 7 of
the roadmap.

- **A1.** TweetEval-emotion replaced by XED as a core corpus (§2.1, §2.5, §4), because
  TweetEval-emotion is derived from SemEval-2018 Task 1.
- **A2.** Prior-work positioning added (§0, Chapter 4). Bostan and Klinger (2018) already
  ran a cross-corpus matrix, so Chapter 4's contribution is restated.
- **A3.** The prediction cache, evaluation-set fan-out, and registries added to the
  repository changes (§2.5). The noise analysis is reworded to match (§2.4).
- **A4.** The hyperparameter-search selection metric named (§2.3, item 3).
- **A5.** The transformer fine-tuning recipe recorded as protocol (§2.3, item 9).
  Chapter 1 gains the study's published numbers as a third column.
- The schedule in §3 is marked as superseded by the roadmap.
