# Thesis Expansion Epic: Implementation Roadmap

**Date:** 2026-11-21
**Status:** Approved 2026-11-21, including the XED swap and spec amendments A1 to A5 (section 7). No implementation has started.
**Spec:** `docs/superpowers/specs/2026-09-25-thesis-expansion-design.md`

The spec says *what* the thesis measures. This document says *what has to be built* to
measure it: the gap between today's code and the spec, the architecture decisions that
close it, and the work split into eleven plans with dependencies and done-criteria.
Each plan gets its own task-level implementation plan (TDD steps, exact code) written
just before it is executed. This roadmap is the index over those plans.

---

## 1. Where things stand (verified 2026-11-21)

- **Design:** committed in September (`57a04ee`, `e520ccd`, `ed4a6ff`) and amended on
  2026-11-21 with the five changes in section 7.
- **Implementation plan:** did not exist before this document.
- **Code:** unchanged since 2026-08-05. The package is 2,968 lines; 62 tests pass.
- **Schedule:** the spec's October and November work (loaders, protocol, new models,
  Chapter 1 re-run) has not started. Fall was unofficial buffer, so the official
  Spring/Summer 2027 plan is intact, but the head start is gone. See section 6.
- **Environment:** local venv is Python 3.14, CPU-only torch. None of the new
  dependencies are installed (codecarbon, sentence-transformers, xgboost, krippendorff,
  nlpaug, anthropic). Wheel availability on 3.14 must be checked at install time.
- **Tracker:** no Linear project or issues exist for this study.

---

## 2. What the code cannot do today

Read from the source, not from the spec. These are the gaps that shape the plans.

| Needed by the spec | Today | Consequence |
|---|---|---|
| Tune thresholds, calibrate, and run paired tests on the same predictions | `run_experiment()` trains, scores at 0.5, and returns only a scorecard row. Per-example probabilities are discarded. | Every post-hoc analysis would need a retrain. This is the central gap. |
| Evaluate one trained model on many test sets (transfer matrix, perturbed sets) | `--test-dataset` takes one target and retrains per target. Models are never saved. | A 6x6 matrix would retrain each model six times per seed. |
| A validation split in the loop | Loaders return it; nothing uses it. | No tuning of any kind is possible without touching test. |
| ~20 model configurations with search spaces | `build_model()` is an if-chain inside `scripts/run_experiment.py`; hyperparameters are constructor defaults. | Does not scale; no place to declare a search space or a route. |
| Six corpora with per-corpus Ekman maps and split policies | Two loaders; `EKMAN_PROJECTORS` is a dict inside the script. | No registry to hang a new corpus on. |
| Shared-emotion macro-F1 for any corpus pair | The Ekman schema always appends neutral. The neutral exclusion is done by hand in `make_crossdataset_figure.py` with a hard-coded set. | The rule exists only for one pair, only in a figure script. |
| Energy, calibrated ECE, threshold mode on the scorecard | `ScorecardRow` and `_SCORECARD_FIELDS` have fixed columns. | New columns must be threaded through row, CSV, and summary. |
| Standard transformer fine-tuning | Hand-rolled loop: constant learning rate, no warmup or decay, batch 16, no validation monitoring. | A reviewer can argue the transformer ceiling is under-tuned. |
| Per-seed subsampling | `--subsample-train` draws with `--seed`, not the loop seed, and mutates the split in place. | Multi-seed matched-size runs silently reuse one draw. |

What already works and should be kept: the `EmotionModel` protocol (new models need
only `fit`, `predict_proba`, `size_mb`), the metrics and bootstrap functions, results
persistence with the stem convention, lazy imports that keep CI network-free and
torch-free, and the `run_m*.sh` batch pattern for the GPU server.

---

## 3. Architecture decisions

These five decisions carry the whole expansion. Everything else follows from them.

**D1. Fit once, predict on many, cache the probabilities.**
The runner is split in two. `fit_and_predict(model, train, eval_sets)` trains once,
then predicts on every named evaluation set: the corpus's own validation and test
splits, every other corpus's test split, and every perturbed test set. Each set's
probabilities are written to disk with a manifest (model, parameters, seed, train
corpus, schema, eval-set name, timings, energy, device, size). A separate `score()`
turns cached predictions into scorecard rows. Threshold tuning, temperature scaling,
paired significance, shared-emotion macro-F1, the transfer matrix, and the framework
all become cheap, repeatable analyses over the cache, runnable on a laptop with no GPU.

**D2. No general model persistence.**
Because a trained model predicts on every evaluation set in the same process, models
do not need to be saved and reloaded. The cost is that evaluation sets must exist
before training; a missing one means a retrain, which on the H200 is minutes. The
transformer tier gets an optional `--save-model` so late-arriving evaluation sets
(for example the noise suite) can be scored without retraining. Compression arms
fine-tune their base model inside `fit()` and then compress, so their training cost
honestly includes the base.

**D3. Registries replace the if-chains.**
A corpus registry (`corpora.py`): one `CorpusSpec` per corpus with its loader, native
labels, Ekman map, single- or multi-label flag, split policy, and stem prefix. A model
registry (`models/registry.py`): one `ModelSpec` per configuration with its builder,
route (S, F, C, ceiling, hosted), search space, and whether it is stochastic. The CLI
becomes a thin layer over both.

**D4. Analyses are pure functions over arrays.**
Threshold tuning, temperature scaling, paired bootstrap with Holm correction, and
shared-label macro-F1 take `(y_true, y_prob)` arrays and return numbers. They are
unit-tested on synthetic arrays in CI with no network and no torch.

**D5. Predictions are artifacts, not source.**
Cached probabilities go under `results/thesis/predictions/`, git-ignored (the full
matrix is on the order of gigabytes). Scorecards, per-emotion CSVs, and figures stay
committed under `results/thesis/` with the existing stem convention. For publication
the prediction cache is archived with a DOI (Plan 11).

`run_experiment()` stays as a thin wrapper over the new runner so the existing 62
tests and the `paper_final` reproduction keep working.

---

## 4. The plans

Size is focused working time: S is up to a day, M is two to four days, L is one to two
weeks. "GPU" means the runs happen on the H200 from a batch script; the code itself is
written and tested locally on CPU.

### Plan 1: Pipeline v2 (M, no GPU) — the foundation
- **Builds:** `corpora.py` (registry, schema resolution moved out of the script),
  `models/registry.py`, `predictions.py` (save/load cache and manifest),
  `fit_and_predict()` and `score()` in `experiment.py`, eval-set fan-out in the CLI,
  the `results/thesis/` tree, and the per-seed subsampling fix.
- **Done when:** the 62 existing tests pass unchanged; a scorecard recomputed from the
  cache equals the in-process one exactly; the new runner matches a verbatim copy of
  the old one exactly; a network-marked regression test reproduces the classical rows
  of `go_native_final_scorecard.csv` within 0.002 on the F1 axes and 0.005 on subset
  accuracy (see the reproducibility finding below).
- **Depends on:** nothing.
- **Status:** built on branch `feature/pipeline-v2`. Task-level plan:
  `docs/superpowers/plans/2026-11-21-pipeline-v2.md`.
- **Reproducibility finding from the build.** The published classical rows do not
  reproduce bit-for-bit on a second machine (up to 0.0005 on F1, 0.0024 on subset
  accuracy). The cause: the TF-IDF vocabulary is capped at 50,000 terms, and on
  GoEmotions that cut falls inside a block of 23,998 terms tied at a count of 2, with
  room for 15,660. scikit-learn breaks those ties with an unstable sort, so the
  rare-term tail of the vocabulary is machine-dependent. The fix changes the features,
  so it is listed under Plan 2, not done here.

### Plan 2: Evaluation protocol (M, no GPU)
- **Builds:** per-label threshold tuning on validation; temperature scaling; paired
  bootstrap deltas with Holm correction; shared-label macro-F1; the eight-configuration
  random search; CodeCarbon energy capture with a no-op fallback; new scorecard columns
  (`threshold_mode`, `ece_calibrated`, `train_energy_wh`, `infer_energy_wh_per_1k`,
  `energy_measured`); five-seed default; transformer loop brought to standard practice
  (batch 32, warmup and linear decay); a deterministic TF-IDF vocabulary cutoff (ties at
  the 50,000-term cap broken by a fixed rule, for example alphabetically), so the
  classical tier's features are identical on every machine; a configuration tag in the
  prediction-cache key, so the search's several fits of one model can share a run
  (today a run with two fits on one cache file is refused before training).
- **Decided (spec §2.3, item 3):** the search selects on validation macro-F1 at
  validation-tuned thresholds, the same procedure for every model, since a fixed 0.5
  would penalize models with poorly scaled probabilities. The transformer recipe is
  spec §2.3, item 9.
- **Done when:** every function is unit-tested on synthetic arrays in CI; a test proves
  the test split is never read during tuning or search; a real run fills the new columns.
- **Depends on:** Plan 1.

### Plan 3: Corpora (M, no GPU, needs network)
- **Builds:** loaders for dair-ai/emotion, ISEAR, MELD (text), XED, and the GoEmotions
  per-rater config; optional EmpatheticDialogues; native label lists and Ekman maps;
  seeded 70/10/20 splits for the two corpora without official ones (ISEAR stratified by
  label, XED by each line's rarest label); the shared-emotion rule in the registry; a
  generated pairwise shared-emotion table; a provenance and license table.
- **XED source:** its Hugging Face repo is script-based and will not load with
  `datasets` 5. Read the tab-separated English file (sentence, then comma-separated
  label numbers 1 to 8) from the Helsinki-NLP/XED GitHub repository instead.
- **Done when:** every corpus loads with train, validation, and test; every label map
  is tested in CI against tiny in-repo fixtures; network-marked smoke tests pass locally.
- **Depends on:** Plan 1. Runs in parallel with Plans 2 and 4.
- **Note:** XED replaces TweetEval-emotion. See section 7, amendment A1.

### Plan 4: Models A, routes S and F (M, no GPU)
- **Builds:** XGBoost in `classical.py`; TextCNN in `deep.py`; `models/embeddings.py`
  with frozen sentence-encoder plus LogReg and GloVe-average plus LogReg; registry
  entries and search spaces for ELECTRA-small and MiniLM fine-tuning; new extras in
  `pyproject.toml`.
- **Done when:** every model passes one shared contract test (output shape and range,
  positive size, same seed gives same output) on a toy corpus; torch-free models run
  in CI, the rest are slow-marked.
- **Depends on:** Plan 1. Runs in parallel with Plans 2 and 3.

### Plan 5: Chapter 1 runs and figures (M, GPU)
- **Builds:** `run_ch1.sh`; Chapter 1 scorecards for three schemas with fixed-0.5 and
  tuned columns, five seeds, paired tests; the paired-delta figure, the energy Pareto,
  and the before/after threshold figure.
- **Done when:** every table of the independent study exists under the new protocol
  and every figure renders from committed CSVs alone.
- **Check on the GPU:** each model now predicts several evaluation sets in a row, and
  the first one absorbs any one-off warm-up. Measure whether that skews
  `predict_latency_ms` for the first set, and if it does, add a warm-up pass before
  timing.
- **Depends on:** Plans 2 and 4.

### Plan 6: Transfer matrix (M, GPU)
- **Builds:** `scripts/run_transfer_matrix.py` (one fit per source corpus, predictions
  on every target); `domain_distance.py` with the four distances; the regression of
  transfer F1 on distance and source size; heatmaps and the distance scatter.
- **Done when:** a full matrix exists per model with a shared-emotion count in every
  cell, and the GoEmotions and SemEval cells show the study's direction of effect.
- **Depends on:** Plans 2 and 3. Can start with the seven existing models.

### Plan 7: Models B, compression arms and extra ceilings (M, GPU)
- **Builds:** `models/compressed.py` (int8 dynamic quantization; 50% magnitude pruning
  with one recovery epoch; sparse-aware size); RoBERTa-base and ModernBERT-base
  entries; the iso-budget tier analysis; the three-frontier figure.
- **Depends on:** Plans 2 and 5.

### Plan 8: Hosted tier (M, needs API access)
- **Builds:** `models/hosted.py` with the open LLM on the H200, the Claude Haiku 4.5
  client, and the Jev client (one Noul question per emotion); committed prompts and
  question templates; an on-disk response cache so reruns cost nothing; cost accounting
  into `cost_usd`; the dollar cap with a stratified subsample.
- **Depends on:** Plans 1 and 2. External: Jev early access, an Anthropic API key.

### Plan 9: Mechanism analyses (M to L, mostly no GPU)
- **9a human ceiling:** rater F1 and Krippendorff's alpha from the per-rater loader.
- **9b calibration:** reliability diagrams from the prediction cache. Nearly free.
- **9c noise:** six perturbations at three intensities, registered as evaluation sets.
  Build the perturbed sets before the final training runs, or use saved checkpoints.
- **9d learning curves:** `scripts/run_learning_curves.py`. GPU. First to be cut.
- **Depends on:** Plans 2, 3, and 5.

### Plan 10: Selection framework (M, no GPU)
- **Builds:** `framework.py` (constraint profiles, feasibility, derivation,
  leave-one-corpus-out regret); `scripts/make_framework.py`; the regret chart and the
  decision-tree render.
- **Depends on:** Plans 6 and 7, since it consumes the complete scorecards.

### Plan 11: Reproducibility and release (S to M) — new, publication-driven
- **Builds:** lock files for both environments (local CPU, server CUDA); one command
  per table and figure; a data provenance and license table, with the SemEval-2018
  community mirror replaced by the official data; a config and seed manifest per
  result; an archived release of results and predictions with a DOI; `CITATION.cff`.
- **Depends on:** the runs it documents. Starts as soon as Plan 5 produces results.

---

## 5. Order, critical path, and milestones

```
Plan 1 ──┬── Plan 2 ──┬── Plan 5 ──┬── Plan 7 ──┐
         ├── Plan 4 ──┘            └── Plan 9   ├── Plan 10
         ├── Plan 3 ──── Plan 6 ────────────────┘
         └── Plan 8                    Plan 11 follows Plans 5 and 6
```

| Milestone | Plans | What it gives |
|---|---|---|
| **M0 Foundation** | 1, 2, 3, 4 | All engineering that needs no GPU. Testable on the laptop. |
| **M1 Publishable core** | 5, 6, 11 | Rigor-fixed replication plus a multi-corpus transfer matrix. Closes all five soft spots in the spec. This is the version of the existing paper that survives review. |
| **M2 Routes** | 7, 8 | The three-routes comparison and the hosted tier (thesis Chapter 2). |
| **M3 Mechanism and framework** | 9, 10 | Thesis Chapters 3 and 4 complete. |

The critical path to M1 is Plan 1, then Plan 2, then Plan 5, with Plans 3 and 6 running
alongside. M1 is deliberately a standalone result: if a publication deadline arrives
before the thesis is done, M1 is the paper and M2 and M3 are the thesis extension.

---

## 6. Revised schedule

The spec assumed Plans 1 to 4 and the Chapter 1 re-run would be done by December. None
of it is. The plans in M0 are well specified and need no GPU, so they can be done now.

| When | Work | Milestone |
|---|---|---|
| Late November to December 2026 | Plans 1 and 2; join the Jev waitlist; thesis proposal draft | |
| January 2027 | Plans 3 and 4; Chapter 1 runs (Plan 5) | M0 |
| February 2027 | Transfer matrix (Plan 6); Plan 11 started | M1 |
| March to April 2027 | Plans 7 and 8; Chapters 1, 2, and 4 written as results land | M2 |
| May to June 2027 | Plan 9 (9a, 9b, 9c first; 9d last) | |
| July 2027 | Plan 10 validation; all figures final; full draft | M3 |
| August 2027 | Revisions and defense preparation | |

If a venue deadline is set, only the M1 date moves; the order does not change.

---

## 7. Spec amendments found during planning

All five were approved and applied to the spec on 2026-11-21 (spec §5 is the log).
They are kept here with their reasons.

**A1. Replace TweetEval-emotion with XED.** TweetEval's emotion task is built from
SemEval-2018 Task 1, which is already a core corpus. Using both would put overlapping
tweets on both sides of a "cross-corpus" cell, and it would make three of six corpora
Twitter. XED (Öhman et al., COLING 2020) is movie subtitles, multi-label, annotated
with Plutchik's eight emotions, so it maps to Ekman-6 exactly as SemEval-2018 does and
covers all six. It also raises the smallest pairwise shared-emotion count from three
to four. It has about 17,500 annotated English lines under CC BY 4.0 and no official
splits. EmpatheticDialogues stays as the optional seventh corpus and the fallback.

**A2. Cite and differentiate from Bostan and Klinger (COLING 2018).** *An Analysis of
Annotated Corpora for Emotion Classification in Text* already ran a cross-corpus
train/test matrix over 14 corpora with a MaxEnt bag-of-words classifier. The transfer
matrix is therefore not new in itself. The contribution of Chapter 4 is narrower and
still open: how transfer differs *across model classes and routes*, whether measurable
domain distance predicts it, and whether a selection rubric derived from it holds on an
unseen corpus. The spec's Chapter 4 framing and the related-work section should say so.

**A3. Add the prediction cache and eval-set fan-out to the spec's repository table.**
Decisions D1, D2, and D5 above are not in the spec's section 2.5.

**A4. Name the hyperparameter-search selection metric** (see Plan 2).

**A5. Record the transformer training-loop change** (batch 32, warmup, linear decay)
as part of the protocol, since it can move the ceiling numbers.

---

## 8. Open questions

1. **Publication target.** Which venue, what deadline, and is it the existing paper or
   the expanded study? This sets the M1 date and whether Plan 11 moves earlier.
2. **ISEAR shame and guilt.** The spec maps both to sadness, matching how GoEmotions
   maps remorse and embarrassment. The stricter alternative is to drop them, as
   anticipation and trust are dropped for SemEval-2018 and XED. A reviewer may prefer
   dropping. Decide before Plan 3.
3. **Tracking.** Should the eleven plans and four milestones be created in Linear?
