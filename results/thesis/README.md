# Thesis results

Everything produced for the thesis expansion lives here, next to `paper_final/`
(the independent study) and `exploratory/` (its archive). The design is in
`docs/superpowers/specs/2026-09-25-thesis-expansion-design.md`.

## Layout

```
results/thesis/
└── <batch>/                          one folder per batch of runs (e.g. ch1/, transfer/)
    ├── <stem>.json                   full record of every scorecard row
    ├── <stem>_scorecard.csv          one row per (model, seed, evaluation set)
    ├── <stem>_per_emotion.csv        per-emotion F1 and support
    ├── <stem>_summary.csv            multi-seed mean and std (only with --seeds)
    └── predictions/<stem>/           per-example probabilities (git-ignored)
        └── <model>__seed<seed>__<corpus>-<split>.npz
```

A batch is written by passing `--out-dir results/thesis/<batch>` to
`scripts/run_experiment.py`. Stems follow the convention documented in
`../paper_final/data/README.md`.

## The scorecard's `dataset` column

Each model is trained once and evaluated on every evaluation set, so one run can
hold several rows per model. The `dataset` column tells them apart:

| Value | Meaning |
|---|---|
| `go_emotions` | trained and tested on GoEmotions (its test split) |
| `go_emotions->sem_eval_2018_task_1` | trained on GoEmotions, tested on SemEval-2018 |
| `go_emotions:validation` | a split other than the test split |

## The prediction cache

Every `.npz` file holds `y_true`, `y_prob`, the label names, and a manifest.
A file is enough to rebuild its scorecard row exactly, so thresholds, calibration
and significance tests never need a retrain, and a batch that dies midway loses
nothing that had finished:

```python
from emotion_classification.pipeline import score_record
from emotion_classification.predictions import list_predictions, load_prediction

for meta in list_predictions("results/thesis/ch1/predictions"):
    if meta["scored"]:                       # the validation record has no row
        row = score_record(load_prediction(meta["path"]))
```

Pass `bootstrap=<n>` to `score_record` to reproduce a run's confidence intervals;
the resample count belongs to the scoring, so it is not stored with the predictions.

### What the manifest records

| Field | Meaning |
|---|---|
| `model`, `model_key`, `route` | the fitted model's name, its registry name, and its route |
| `params` | the hyperparameters it was built with (`head.C`, `vectorizer.max_features`, `epochs`, ...) |
| `seed`, `train_corpus`, `train_split`, `schema`, `features` | what was trained, on what |
| `n_train`, `subsample_train`, `limit_train`, `limit_eval` | how much data |
| `eval_set`, `dataset`, `test_dataset`, `n_eval` | what was predicted; `dataset` is the scorecard label |
| `scored` | `false` for the training corpus's validation split, cached only for tuning |
| `train_seconds`, `predict_seconds`, `model_size_mb`, `device` | the efficiency measurements |
| `versions`, `created_utc` | Python and library versions, and when the fit ran |

### Things to know

- **One file per (run name, model, seed, evaluation set).** The key has no
  hyperparameter part. A run in which two fits would land on the same file is
  refused before training; run different configurations of one model under
  different run names.
- **Latency order.** Evaluation sets are predicted in the order given, each timed
  on its own, and the validation split always goes last. The first set includes any
  one-off warm-up, so list the set whose latency you report first.
- **Reusing a run name** overwrites matching files and leaves the others in place.
  `created_utc` tells old records from new.
- The cache is git-ignored because of its size. It is regenerable from the batch
  scripts and is archived separately for publication.
