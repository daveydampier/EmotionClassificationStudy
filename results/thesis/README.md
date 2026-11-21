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

Every `.npz` file holds `y_true`, `y_prob`, the label names, and a manifest
(model, seed, training corpus, schema, evaluation set, timings, model size,
device). Scores are recomputed from these files exactly, so thresholds,
calibration and significance tests never need a retrain:

```python
from emotion_classification.experiment import score
from emotion_classification.predictions import list_predictions, load_prediction

for meta in list_predictions("results/thesis/ch1/predictions"):
    row = score(load_prediction(meta["path"]))
```

The training corpus's validation split is cached alongside the scored sets but
has no scorecard row; it exists for tuning. The cache is git-ignored because of
its size. It is regenerable from the batch scripts and is archived separately
for publication.
