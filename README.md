# Emotion Classification Study

Independent-study coding project for **text-based emotion classification**. The goal is to
go beyond the usual Accuracy/Precision/Recall/F1 comparisons and evaluate classical ML,
deep-learning, and transformer models across additional real-world dimensions
(efficiency, robustness, calibration, human-alignment, cost) — ultimately producing a
normalized multi-dimension **"deployment scorecard."**

This repo holds the experimentation code. Deep-learning and transformer models are built
with **PyTorch** + the **HuggingFace** stack (`transformers`, `datasets`); classical
models use **scikit-learn**.

> **Status:** basic project scaffold. Model tiers, dataset loaders, and the scorecard
> metrics module are not implemented yet.

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

> ⚠️ **This repo lives in a OneDrive-synced folder.** OneDrive's Files-On-Demand
> reparse points break `python -m venv` *inside* the repo, and you wouldn't want a
> multi-GB venv syncing to the cloud anyway. Create the venv **outside** OneDrive.

```bash
# 1. Create a virtual environment OUTSIDE OneDrive
#    Git Bash:
python -m venv "$HOME/venvs/EmotionClassificationStudy"
#    PowerShell:
#    python -m venv $env:USERPROFILE\venvs\EmotionClassificationStudy

# 2. Activate it
#    PowerShell:
& $env:USERPROFILE\venvs\EmotionClassificationStudy\Scripts\Activate.ps1
#    Git Bash:
source "$HOME/venvs/EmotionClassificationStudy/Scripts/activate"

# 3. Install dependencies (run from the repo root)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3b. (Optional) Replace CPU torch with a CUDA build — pick the cu### for your driver:
#     pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Verify the environment (library versions + GPU visibility)
python scripts/check_env.py
```

> Optional: enabling Windows **Developer Mode** lets the HuggingFace cache use
> symlinks (more disk-efficient); without it, downloads still work but are copied.

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
│   └── loaders.py            # normalized dataset loaders (GoEmotions, SemEval)
├── data/
│   ├── raw/                  # downloaded datasets (git-ignored)
│   └── processed/            # cleaned / harmonized data (git-ignored)
├── notebooks/                # exploratory & visualization notebooks
├── scripts/
│   └── check_env.py          # environment / GPU sanity check
├── tests/                    # pytest suite
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

---

## Model tiers (planned)

| Tier | Library | Examples |
|---|---|---|
| Classical | scikit-learn | SVM, Naive Bayes, Random Forest (+ TF-IDF / SBERT features) |
| Deep learning | PyTorch | BiLSTM, CNN (+ GloVe embeddings) |
| Transformer | PyTorch + HuggingFace | BERT, RoBERTa (fine-tuned) |
| LLM (optional) | API | prompted GPT / Claude / Gemini |

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
