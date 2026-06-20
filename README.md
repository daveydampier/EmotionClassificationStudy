# Emotion Classification Study

Independent-study coding project for **text-based emotion classification**. The goal is to
go beyond the usual Accuracy/Precision/Recall/F1 comparisons and evaluate classical ML,
deep-learning, and transformer models across additional real-world dimensions
(efficiency, robustness, calibration, human-alignment, cost) — ultimately producing a
normalized multi-dimension **"deployment scorecard."**

This repo holds the experimentation code. Deep-learning and transformer models are built
with **PyTorch** + the **HuggingFace** stack (`transformers`, `datasets`); classical
models use **scikit-learn**.

> **Status:** end-to-end pipeline working. Dataset loaders, label harmonization,
> the metrics + scorecard modules, and all three model tiers (classical / BiLSTM /
> transformer) run from `scripts/run_experiment.py`. Next up: visualization
> (radar / Pareto plots), the robustness & human-alignment axes, and BRIGHTER.

---

## Requirements

- **Python 3.9–3.14.** The full stack (torch, transformers, datasets, scikit-learn, …)
  has wheels for **Python 3.14**, so the interpreter already on this machine works — no
  downgrade needed.
- **GPU is optional.** The default `torch` wheel on Windows is **CPU-only**, which is fine
  for the classical and small-DL tiers. For NVIDIA/CUDA acceleration (useful for
  BERT/RoBERTa fine-tuning), install torch from the PyTorch index — see step 3b below.

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
│   ├── metrics.py            # predictive + calibration (ECE) metric contract
│   ├── scorecard.py          # deployment scorecard: rows + [0,1] normalization
│   ├── experiment.py         # seed + time + score a model into a scorecard row
│   └── models/               # one interface, three tiers
│       ├── base.py           # EmotionModel protocol (fit/predict_proba/size_mb)
│       ├── classical.py      # TF-IDF + LogReg / LinearSVM (scikit-learn)
│       ├── deep.py           # BiLSTM (PyTorch)
│       └── transformer.py    # BERT / RoBERTa fine-tuning (HuggingFace)
├── data/
│   ├── raw/                  # downloaded datasets (git-ignored)
│   └── processed/            # cleaned / harmonized data (git-ignored)
├── notebooks/                # exploratory & visualization notebooks
├── scripts/
│   ├── check_env.py          # environment / GPU sanity check
│   └── run_experiment.py     # CLI: run tiers on a dataset, print scorecard
├── tests/                    # pytest suite
├── .github/workflows/ci.yml  # CI: no-network tests on push / PR
├── pyproject.toml            # deps + extras + packaging + pytest config
├── requirements.lock.txt     # exact pins from `pip freeze` (reproducibility)
├── .gitignore
└── README.md
```

---

## Running experiments

The whole pipeline — load → project to Ekman-6 → train → score — runs from one CLI:

```bash
# Classical tier on GoEmotions (fast; caps train rows for a quick smoke run)
python scripts/run_experiment.py --models tfidf_logreg tfidf_linearsvm \
    --dataset go_emotions --limit-train 4000 --limit-test 1500

# Add the BiLSTM (PyTorch) tier
python scripts/run_experiment.py --models tfidf_logreg bilstm --dataset go_emotions

# Transformer tier (downloads weights; slow on CPU)
python scripts/run_experiment.py --models hf_distilbert --dataset go_emotions \
    --limit-train 2000 --limit-test 1000
```

Each run prints a **raw scorecard** (macro/micro-F1, subset accuracy, ECE
calibration, train seconds, per-sample latency, model size MB) and, for >1 model,
a **normalized** table where 1.0 is best on every axis (lower-is-better axes are
inverted) — the input to radar / Pareto plots.

To add a model, implement the `EmotionModel` interface
(`emotion_classification/models/base.py`) and register it in
`scripts/run_experiment.py`; metrics, timing, and the scorecard come for free.

---

## Model tiers

| Tier | Library | Implemented | Examples |
|---|---|---|---|
| Classical | scikit-learn | ✅ `models/classical.py` | TF-IDF + LogReg / LinearSVM (NB, RF, SBERT to add) |
| Deep learning | PyTorch | ✅ `models/deep.py` | BiLSTM (learned embeddings); CNN / GloVe to add |
| Transformer | PyTorch + HuggingFace | ✅ `models/transformer.py` | DistilBERT default; BERT / RoBERTa via `--models hf_<name>` |
| LLM (optional) | API | ⏳ | prompted GPT / Claude / Gemini |

## Datasets

Loaders live in `emotion_classification/loaders.py` and normalize each source to a
common `(text, labels)` form; `labels.py` projects labels into a shared schema.

| Dataset | Role | HF source | Status |
|---|---|---|---|
| **GoEmotions** (27 + neutral) | Anchor | `google-research-datasets/go_emotions` (`simplified`) | ✅ loader works |
| **SemEval-2018 Task 1 E-c** (11, multi-label) | Co-occurrence analysis | `vibhorag101/sem_eval_2018_task_1_english_cleaned_labels` ⚠️ community mirror, English-only | ✅ loader works |
| **SemEval-2025 / BRIGHTER** | Cross-lingual transfer target | _TBD — confirm HF path_ | ⏳ stubbed |

> Note: the official `sem_eval_2018_task_1` repo is script-based and unsupported by
> `datasets` >= 5, so we use a script-free Parquet mirror. Verify its provenance
> before citing results, or substitute the official data.

The **label-harmonization map** (GoEmotions 27 → Ekman 6 / sentiment; SemEval-2018 11
→ Ekman 6) is implemented in `labels.py` and tested in `tests/test_labels.py`.
