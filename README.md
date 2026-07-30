# Emotion Classification Study

Independent-study project for **text-based emotion classification**. It goes beyond the
usual Accuracy/Precision/Recall/F1 comparison to evaluate classical ML, deep-learning, and
transformer models across real-world deployment dimensions — training cost, inference
throughput, model size, calibration, per-emotion behaviour, and cross-dataset robustness —
collected into a normalized multi-dimension **"deployment scorecard."** The guiding question
is *the cost of going lightweight*: what a small model actually sacrifices, and where.

Deep-learning and transformer models are built with **PyTorch** + the **HuggingFace** stack
(`transformers`, `datasets`); classical models use **scikit-learn**.

> **Status: complete.** Seven models across three tiers were run on GoEmotions
> (native-27, Ekman-6, sentiment-3), with multi-seed and bootstrap uncertainty, a
> cross-dataset transfer test (GoEmotions → SemEval-2018), and a transformer
> training-budget check. All results and figures are committed under
> [`results/`](results/) (see [`results/README.md`](results/README.md) for a file key).
> For the engineering reference, see [README_DETAILED.md](README_DETAILED.md).

---

## Requirements

- **Python 3.9–3.14.** The full stack (torch, transformers, datasets, scikit-learn, …)
  ships wheels through Python 3.14; `pyproject.toml` pins `>=3.9,<3.15`.
- **GPU is optional.** The default `torch` wheel on Windows is **CPU-only**, which is fine
  for the classical and small-DL tiers. For NVIDIA/CUDA acceleration (used here for
  BERT/DistilBERT fine-tuning on an H200), install torch from the PyTorch index — see
  step 3b below.

---

## Setup

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
#    PowerShell:
.venv\Scripts\Activate.ps1
#    Git Bash:
source .venv/Scripts/activate

# 3. Install the package + dependencies (editable). Extras are defined in
#    pyproject.toml; ".[all]" is the full stack, or pick what you need:
python -m pip install --upgrade pip
pip install -e ".[all]"
#    Lighter options:
#      pip install -e ".[transformers,viz,dev]"   # full modeling stack, no JupyterLab
#      pip install -e ".[dev]"                     # core + classical tier + pytest only

# 3b. (Optional) Replace CPU torch with a CUDA build — pick the cu### for your driver:
#     pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Verify the environment (library versions + GPU visibility)
python scripts/check_env.py
```

> Optional: enabling Windows **Developer Mode** lets the HuggingFace cache use
> symlinks (more disk-efficient); without it, downloads still work but are copied.
>
> ⚠️ **Windows long paths:** installing `jupyter` pulls JupyterLab, whose deeply
> nested extension files can exceed the 260-char `MAX_PATH` limit and abort the
> install. Either enable long paths (`Set-ItemProperty
> 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' LongPathsEnabled 1`, admin)
> or skip notebooks: `pip install -e ".[transformers,viz,dev]"` installs the full
> modeling stack without JupyterLab.

After a successful install, freeze exact versions for reproducibility:

```bash
pip freeze > requirements.lock.txt
```

---

## Project layout

```
EmotionClassificationStudy/
├── emotion_classification/   # main package
│   ├── __init__.py
│   ├── config.py             # canonical project paths
│   ├── labels.py             # label schemas + cross-dataset harmonization
│   ├── loaders.py            # normalized dataset loaders (GoEmotions, SemEval)
│   ├── preprocessing.py      # project native labels -> multi-hot target arrays
│   ├── metrics.py            # predictive + calibration (ECE) + bootstrap CIs
│   ├── scorecard.py          # deployment scorecard: rows + [0,1] normalization
│   ├── experiment.py         # seed + time + score a model into a scorecard row
│   ├── results.py            # persist/load scorecards (CSV/JSON) + multi-seed summary
│   ├── viz.py                # the figures (Pareto, heatmap, violin, F1-vs-frequency)
│   └── models/               # one interface, three tiers
│       ├── base.py           # EmotionModel protocol (fit/predict_proba/size_mb)
│       ├── classical.py      # NB / LogReg / LinearSVM / RandomForest (TF-IDF, BoW)
│       ├── deep.py           # BiLSTM (PyTorch)
│       └── transformer.py    # BERT / DistilBERT fine-tuning (HuggingFace)
├── data/
│   ├── raw/                  # downloaded datasets (git-ignored)
│   └── processed/            # cleaned / harmonized data (git-ignored)
├── scripts/
│   ├── check_env.py          # environment / GPU sanity check
│   ├── run_experiment.py     # CLI: run tiers on a dataset+schema, persist scorecard
│   └── make_figures.py       # CLI: turn persisted CSVs into PNG figures
├── results/                  # committed run outputs (CSV/JSON) + figures/ + README key
├── run_m2.sh, run_m3.sh      # GPU-server batch scripts (the runs behind results/)
├── tests/                    # pytest suite (62 tests)
├── .github/workflows/ci.yml  # CI: no-network tests on push / PR
├── pyproject.toml            # deps + extras + packaging + pytest config
├── requirements.lock.txt     # exact pins from `pip freeze` (reproducibility)
├── README.md                 # this file
└── README_DETAILED.md        # engineering reference (module-by-module)
```

---

## Running experiments

The whole pipeline — load → project to a schema → train → score → persist — runs from one
CLI. Models are chosen with `--models` (`naive_bayes`, `logreg`, `linsvm`,
`random_forest`, `bilstm`, `hf_<name>`) and the classical feature backend with
`--features` (`tfidf` or `bow`):

```bash
# Classical tier on GoEmotions, native 27-class (caps rows for a quick smoke run)
python scripts/run_experiment.py --models naive_bayes logreg linsvm random_forest \
    --dataset go_emotions --schema native --limit-train 4000 --limit-test 1500

# Add the BiLSTM (PyTorch) tier
python scripts/run_experiment.py --models logreg bilstm --dataset go_emotions

# Transformer tier (downloads weights; slow on CPU, use a GPU)
python scripts/run_experiment.py --models hf_distilbert --dataset go_emotions \
    --limit-train 2000 --limit-test 1000

# Turn a run's persisted CSVs into figures (no models re-run)
python scripts/make_figures.py --results-dir results --name go_emotions_native_tfidf
```

Additional analyses (all in one CLI): `--seeds 42 43 44` for multi-seed mean±std,
`--bootstrap 1000` for test-set F1 confidence intervals, `--epochs N` for the
transformer training-budget check, and `--test-dataset sem_eval_2018_task_1` (with
`--schema ekman6`) for cross-dataset transfer. The exact batches behind the committed
results are `run_m2.sh` / `run_m3.sh`. See [README_DETAILED.md](README_DETAILED.md) for
the full flag reference and typical workflows.

Each run prints a **raw scorecard** (macro/micro-F1, subset accuracy, ECE, train
seconds, per-sample latency, throughput, model size MB) and, for >1 model, a
**normalized** table where 1.0 is best on every axis (lower-is-better axes inverted) —
the input to the Pareto plots. To add a model, implement the `EmotionModel` interface
(`emotion_classification/models/base.py`) and register it in
`scripts/run_experiment.py`; metrics, timing, and the scorecard come for free.

---

## Model tiers

| Tier | Library | Module | Models used in the study |
|---|---|---|---|
| Classical (lightweight) | scikit-learn | `models/classical.py` | Naive Bayes (ComplementNB), Logistic Regression, Linear SVM, Random Forest — each on TF-IDF and BoW |
| Deep learning | PyTorch | `models/deep.py` | BiLSTM (learned embeddings) |
| Transformer (reference) | PyTorch + HuggingFace | `models/transformer.py` | BERT-base, DistilBERT (`--models hf_<name>` accepts any BERT/RoBERTa checkpoint) |

## Datasets

Loaders live in `emotion_classification/loaders.py` and normalize each source to a
common `(text, labels)` form; `labels.py` projects labels into a shared schema.

| Dataset | Role | HF source | Status |
|---|---|---|---|
| **GoEmotions** (27 + neutral) | Primary corpus (all in-domain results) | `google-research-datasets/go_emotions` (`simplified`) | ✅ used |
| **SemEval-2018 Task 1 E-c** (11, multi-label) | Cross-dataset transfer target (shared Ekman-6) | `vibhorag101/sem_eval_2018_task_1_english_cleaned_labels` ⚠️ community mirror, English-only | ✅ used |
| **SemEval-2025 / BRIGHTER** | Cross-lingual transfer | _TBD — confirm HF path_ | ⏳ future work (loader stubbed, not used) |

> Note: the official `sem_eval_2018_task_1` repo is script-based and unsupported by
> `datasets` >= 5, so we use a script-free Parquet mirror. Its provenance is flagged in
> the paper; substitute the official data to reproduce against the canonical source.
> The **BRIGHTER** row is a wired-in extension point (`load_brighter` raises
> `NotImplementedError`) for a planned cross-lingual experiment — it is **not** part of
> the results in this repo.

The **label-harmonization map** (GoEmotions 27 → Ekman 6 / sentiment; SemEval-2018 11
→ Ekman 6) is implemented in `labels.py` and tested in `tests/test_labels.py`.
