# Emotion Classification Study — Detailed Codebase Documentation

This document is the engineering reference for the project. For the research
framing and quick-start setup, see [README.md](README.md); for the study
proposal, see [docs/proposal.md](docs/proposal.md).

The study compares **classical, deep-learning, and transformer** models for
text emotion classification not only on accuracy but on **deployment cost**
(training time, model size, inference latency, calibration) — with a focus on
whether lightweight models lose accuracy *uniformly* across emotions or
*disproportionately on rare ones*.

---

## Table of Contents
1. [Architecture & data flow](#architecture--data-flow)
2. [Directory layout](#directory-layout)
3. [Core concepts](#core-concepts)
4. [Module reference](#module-reference)
5. [Command-line scripts](#command-line-scripts)
6. [Label schemas](#label-schemas)
7. [Metrics](#metrics)
8. [Persisted results format](#persisted-results-format)
9. [Typical workflows](#typical-workflows)
10. [Extending the study](#extending-the-study)
11. [Testing & CI](#testing--ci)

---

## Architecture & data flow

The pipeline is a single linear flow, and every model tier passes through the
exact same stages so results are directly comparable:

```
loaders.py            preprocessing.py         models/*.py        experiment.py         results.py / viz.py
──────────            ────────────────         ───────────        ─────────────         ───────────────────
HF dataset  ──►  EmotionDataset  ──►  PreparedSplit (texts,  ──►  EmotionModel  ──►  ScorecardRow  ──►  CSV/JSON + figures
(GoEmotions)     (native labels)      multi-hot Y over a          (.fit/.predict_     (metrics +
                                      target label schema)         proba)              efficiency +
                                                                                       per-emotion)
```

1. **Load** a dataset to a normalized `EmotionDataset` (native label names).
2. **Project** native labels into a target **schema** (native 27-class, Ekman-6,
   or sentiment-3) and build a multi-hot `PreparedSplit`.
3. **Train/evaluate** any model implementing the `EmotionModel` interface.
4. **Score** into a `ScorecardRow` (predictive + calibration + efficiency +
   per-emotion breakdown + device).
5. **Persist** to CSV/JSON, then **visualize** from the persisted files.

Heavy dependencies (torch, transformers, matplotlib) are imported lazily inside
the modules that need them, so `import emotion_classification` stays light and
the classical tier runs without the deep-learning stack installed.

---

## Directory layout

```
emotion_classification/       # main package
├── __init__.py               # version only; no heavy imports
├── config.py                 # canonical filesystem paths
├── labels.py                 # label schemas + cross-dataset harmonization maps
├── loaders.py                # HF dataset loaders -> EmotionDataset
├── preprocessing.py          # project labels -> multi-hot PreparedSplit; class frequency
├── metrics.py                # predictive + calibration + per-emotion metrics
├── scorecard.py              # ScorecardRow + Scorecard (normalization, markdown)
├── experiment.py             # seed + time + score a model -> ScorecardRow
├── results.py                # persist/load scorecards (CSV/JSON)
├── viz.py                    # the five figures (matplotlib/seaborn)
└── models/
    ├── base.py               # EmotionModel protocol + BaseEmotionModel
    ├── classical.py          # NB / LogReg / LinearSVM / RandomForest (BoW/TF-IDF)
    ├── deep.py               # BiLSTM (PyTorch)
    └── transformer.py        # BERT/RoBERTa fine-tuning (HuggingFace)

scripts/
├── check_env.py              # environment / GPU sanity check
├── run_experiment.py         # CLI: run tiers on a dataset+schema, persist scorecard
└── make_figures.py           # CLI: turn persisted CSVs into PNG figures

tests/                        # pytest suite (see Testing & CI)
docs/                         # proposal + writing scaffold
```

---

## Core concepts

### Label schema & harmonization (`labels.py`)
Datasets use different taxonomies. `labels.py` defines each native taxonomy and
**projection maps** into shared, coarser schemas so results are comparable. The
target schemas are **Ekman-6 (+neutral)** and **sentiment-3 (+neutral)**;
"native" means the dataset's own labels (GoEmotions = 27 + neutral). Projections
are pure functions like `goemotions_to_ekman(labels) -> list[str]`.

### PreparedSplit & multi-hot (`preprocessing.py`)
A `PreparedSplit` holds `texts: list[str]` and `Y: np.ndarray` — a
`(n_samples, n_labels)` multi-hot `{0,1}` matrix over a fixed `label_names`
vocabulary. This is the single input format every model consumes. Multi-label
throughout (a comment can carry several emotions).

### The `EmotionModel` interface (`models/base.py`)
Any model that implements four methods drops into the pipeline:
`fit(texts, Y)`, `predict_proba(texts) -> ndarray`, `predict(texts, threshold)`,
and `size_mb()`. It's a `typing.Protocol`, so classes need not subclass it —
they just need matching methods. Keeping probabilities (not just hard labels) in
the contract is deliberate: calibration (ECE) needs them.

### The deployment scorecard (`scorecard.py`)
A `ScorecardRow` records every axis for one (model, dataset) run. `Scorecard`
collects rows and can `normalized()` every axis to `[0,1]` where **1 = best**
(lower-is-better axes are inverted) — the input to radar/Pareto plots.

---

## Module reference

### `config.py`
Canonical paths derived from the package location so code works regardless of
CWD: `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`,
`MODELS_DIR`, `NOTEBOOKS_DIR`, and `ensure_dirs()`.

### `labels.py`
- Taxonomies: `GOEMOTIONS` (28, neutral last), `EKMAN` (6), `SENTIMENT` (3),
  `SEMEVAL2018` (11), `BRIGHTER`, `NEUTRAL`.
- Group maps: `EKMAN_GROUPS`, `SENTIMENT_GROUPS`, `SEMEVAL2018_TO_EKMAN`.
- Projection functions: `goemotions_to_ekman`, `goemotions_to_sentiment`,
  `semeval2018_to_ekman`, and the generic `map_labels(labels, mapping)`.
- Unknown labels raise `KeyError` (typos can't pass silently). Labels mapping to
  `None` (no equivalent in the target) are dropped.

### `loaders.py`
- `Example(text, labels)` and `EmotionDataset(name, label_names, splits, multi_label)`.
- `load_goemotions(config="simplified")` — 27 emotions + neutral, Reddit.
- `load_semeval2018(language="english")` — 11 labels, tweets (script-free mirror).
- `load_brighter(...)` — stubbed (`NotImplementedError`; HF path TBD).
- `LOADERS` — name → loader registry. `datasets` is imported lazily; loaders hit
  the network on first use (HF cache thereafter).

### `preprocessing.py`
- `to_multihot(label_lists, label_names) -> ndarray`.
- `prepare_split(examples, label_names, projector=None, drop_empty=True) -> PreparedSplit`.
- `prepare_dataset(ds, label_names, projector=None) -> {split: PreparedSplit}`.
- `label_support(split) -> {label: positive_count}` — class frequency, the x-axis
  of the core per-emotion figure.
- `Projector` type: `Callable[[list[str]], list[str]]`.

### `metrics.py` (numpy + scikit-learn only)
- `classification_metrics(y_true, y_prob, threshold)` → macro/micro-F1,
  macro precision/recall, `subset_accuracy` (exact-match).
- `expected_calibration_error(y_true, y_prob, n_bins)` → ECE.
- `per_label_metrics(y_true, y_prob, label_names, threshold)` →
  `{label: {f1, precision, recall, support}}` (backbone of RQ3).
- `per_label_f1(...)` → `{label: f1}` convenience.
- `evaluate(...)` → predictive metrics + ECE in one dict.
- `binarize(y_prob, threshold)`.

### `scorecard.py`
- `HIGHER_IS_BETTER` — axis → direction map (drives normalization).
- `ScorecardRow` fields: `model`, `dataset`; `macro_f1`, `micro_f1`,
  `subset_accuracy`; `ece`; `train_seconds`, `predict_latency_ms`, `throughput`
  (samples/sec), `model_size_mb`, `cost_usd`; `device`; `per_label_f1`,
  `per_label_support`; `meta` (schema, features, n_train/test/labels, seed, …).
- `Scorecard`: `add`, `axes`, `normalized()`, `to_dataframe()`, `to_markdown()`.
  Non-axis fields (`device`, per-emotion dicts) never leak into normalization.

### `models/`
- **`base.py`** — `EmotionModel` protocol + `BaseEmotionModel` (supplies a default
  `predict` from `predict_proba`).
- **`classical.py`** — `NaiveBayes` (ComplementNB, imbalance-robust), `LogisticReg`,
  `LinearSVM` (calibrated for probabilities), `RandomForest`. Each takes
  `features="tfidf"|"bow"`; the backend is encoded in `.name` (e.g.
  `naive_bayes_bow`). Shared plumbing: `_VectorizedClassifier` +
  `_BinaryRelevance` (one estimator per label; tolerates constant label columns).
- **`deep.py`** — `BiLSTMClassifier` (whitespace vocab, learned embeddings,
  BiLSTM, BCEWithLogitsLoss). Torch imported lazily.
- **`transformer.py`** — `TransformerClassifier` (HF
  `AutoModelForSequenceClassification`, multi-label; DistilBERT default, any
  BERT/RoBERTa via `model_name`).

### `experiment.py`
- `set_seed(seed)` — seeds Python/NumPy/(torch if present).
- `run_experiment(model, train, test, *, dataset_name, threshold=0.5, seed=42,
  cost_usd=None, extra_meta=None) -> ScorecardRow` — seeds, times fit, times
  inference (→ per-sample latency), scores predictive + calibration + per-emotion,
  measures `size_mb()`, and records the device.
- `_detect_device(model)` → `"cpu"` / `"cuda:0"` (reads a torch model's
  parameters; classical models report CPU). Makes the CPU/GPU timing-fairness
  caveat visible in every row.

### `results.py` (stdlib only)
- `save_results(card, out_dir, name) -> {json, scorecard, per_emotion}` writes
  `<name>.json` (full record), `<name>_scorecard.csv` (flat headline metrics),
  `<name>_per_emotion.csv` (long: model × emotion → F1 + support).
- `load_results(json_path) -> list[dict]`.

### `viz.py` (matplotlib/seaborn/pandas)
- `pareto_scatter(scorecard_df, x="train_seconds", y="macro_f1")`.
- `per_emotion_heatmap(per_emotion_df)` — emotion × model, ordered by frequency.
- `per_emotion_violin(per_emotion_df)` — F1 distribution per model.
- `f1_vs_frequency(per_emotion_df)` — **the core figure** (log-frequency x-axis).
- `granularity_bars(scorecard_df, metric)` — across schemas (needs a multi-schema df).
- `save_all(...)`, `load_scorecard(csv)`, `load_per_emotion(csv)`, `SCHEMA_ORDER`.
- Each figure function returns a Matplotlib `Figure`. Not imported by the package
  `__init__` (keeps library import light).

---

## Command-line scripts

### `scripts/check_env.py`
Prints Python + key library versions and PyTorch CUDA visibility. Exit non-zero
if a core import fails.

### `scripts/run_experiment.py`
Runs one or more models on a dataset+schema, prints a scorecard, and persists it.

| Flag | Default | Meaning |
|---|---|---|
| `--models` | `logreg` | `naive_bayes`, `logreg`, `linsvm`, `random_forest`, `bilstm`, `hf_<name>` |
| `--dataset` | `go_emotions` | any key in `loaders.LOADERS` |
| `--schema` | `native` | `native`, `ekman6`, `sentiment3` |
| `--features` | `tfidf` | `tfidf`, `bow` (classical tier) |
| `--train-split` / `--test-split` | `train` / `test` | dataset split names |
| `--limit-train` / `--limit-test` | none | cap rows (fast smoke runs) |
| `--seed` | `42` | RNG seed |
| `--out-dir` | `results` | where CSV/JSON are written |
| `--run-name` | `<dataset>_<schema>_<features>` | output basename |
| `--no-save` | off | skip writing results |

### `scripts/make_figures.py`
Turns a run's persisted CSVs into PNGs (no models re-run).

| Flag | Default | Meaning |
|---|---|---|
| `--results-dir` | `results` | directory holding the CSVs |
| `--name` | *(required)* | run basename, e.g. `go_emotions_native_tfidf` |
| `--out-dir` | `<results-dir>/figures` | where PNGs are written |

---

## Label schemas

| `--schema` | Target labels | Projector | Notes |
|---|---|---|---|
| `native` | dataset's own (GoEmotions = 27 + neutral) | none | preserves rare-emotion signal for RQ3 |
| `ekman6` | anger, disgust, fear, joy, sadness, surprise (+ neutral) | `goemotions_to_ekman` / `semeval2018_to_ekman` | coarser |
| `sentiment3` | positive, negative, ambiguous (+ neutral) | `goemotions_to_sentiment` | GoEmotions only |

Running the same models across all three is the **granularity sweep** (secondary
analysis): does the lightweight penalty shrink as the taxonomy coarsens?

---

## Metrics

- **macro-F1** — unweighted mean of per-label F1 (rewards rare-class competence).
- **micro-F1** — pools all label decisions (dominated by frequent labels).
- **subset_accuracy** — fraction of examples whose predicted label set exactly
  equals the true set (strict multi-label accuracy).
- **ECE** — Expected Calibration Error over all label-probability scores
  (calibration; lower = better).
- **per-emotion F1 + support** — F1 per label plus its positive count (class
  frequency) in the eval set — feeds the heatmap, violin, and core scatter.

Efficiency axes (`train_seconds`, `predict_latency_ms`, `throughput`
(samples/sec), `model_size_mb`) are recorded per run. Because the lightweight tier
runs on CPU and the transformer on GPU, timing/latency/throughput are **only
comparable within the same `device`** — hence the device is recorded on every row,
and **model size (MB) is the hardware-independent memory proxy**. Throughput is the
inverse of latency, reported explicitly as a named RQ1 efficiency metric.

---

## Persisted results format

`save_results(card, out_dir, name)` writes three files:

- **`<name>.json`** — list of full `ScorecardRow` dicts (everything, incl.
  `meta`, `device`, per-emotion dicts). Round-trips via `load_results`.
- **`<name>_scorecard.csv`** — columns: `model, dataset, schema, features,
  device, macro_f1, micro_f1, subset_accuracy, ece, train_seconds,
  predict_latency_ms, throughput, model_size_mb, cost_usd, n_train, n_test,
  n_labels, seed`.
- **`<name>_per_emotion.csv`** — long format: `model, dataset, schema, seed,
  emotion, f1, support`. This is the shape the per-emotion figures consume.

`results/` is git-ignored (outputs are regenerable).

---

## Typical workflows

```bash
# 1. Lightweight tier on full GoEmotions, native 27-class -> persisted results
python scripts/run_experiment.py \
    --models naive_bayes logreg linsvm random_forest \
    --dataset go_emotions --schema native --run-name go_native

# 2. Figures from those results
python scripts/make_figures.py --results-dir results --name go_native

# 3. Granularity sweep (run each schema, then combine for the granularity figure)
python scripts/run_experiment.py --models logreg --schema ekman6 --run-name go_ekman6
python scripts/run_experiment.py --models logreg --schema sentiment3 --run-name go_sent3

# 4. Transformer reference (needs GPU for reasonable speed)
python scripts/run_experiment.py --models hf_distilbert --schema native \
    --limit-train 2000 --limit-test 1000 --run-name go_native_distilbert
```

To combine multiple runs into one scorecard (e.g. for the granularity figure),
load their JSONs and rebuild a `Scorecard`:

```python
from emotion_classification.results import load_results, save_results
from emotion_classification.scorecard import Scorecard, ScorecardRow
rows = load_results("results/go_native.json") + load_results("results/go_ekman6.json")
save_results(Scorecard([ScorecardRow(**r) for r in rows]), "results", "combined")
```

---

## Extending the study

**Add a classical model:** subclass `_VectorizedClassifier` in `classical.py`
with any scikit-learn estimator exposing `predict_proba`, set a `name`, and
register a short name in `run_experiment.py`'s `CLASSICAL_BUILDERS`.

**Add any model tier:** implement the four `EmotionModel` methods
(`fit`, `predict_proba`, `predict`, `size_mb`); metrics, timing, device
detection, and the scorecard come for free via `run_experiment`.

**Add a dataset:** write a loader returning an `EmotionDataset`, register it in
`loaders.LOADERS`, and add a projector (or reuse `native`) in
`run_experiment.resolve_schema`.

---

## Testing & CI

- `pytest -m "not network and not slow"` runs the offline suite (no HF downloads,
  no long training). Markers: `network` (HF Hub), `slow` (training).
- Tests avoid torch/transformers where possible; viz tests `importorskip` the
  plotting stack so a lean install still passes.
- **CI** (`.github/workflows/ci.yml`) installs `.[dev,viz]` + scikit-learn and
  runs the offline suite on Python 3.11 / 3.12.

Setup and the Windows long-path caveat are covered in [README.md](README.md).
```
