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

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
#    PowerShell:
.venv\Scripts\Activate.ps1
#    Git Bash:
source .venv/Scripts/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3b. (Optional) Replace CPU torch with a CUDA build — pick the cu### for your driver:
#     pip install torch --index-url https://download.pytorch.org/whl/cu124

# 4. Verify the environment (library versions + GPU visibility)
python scripts/check_env.py
```

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

## Planned datasets

| Dataset | Role | Notes |
|---|---|---|
| **GoEmotions** (27 + neutral) | Anchor | Reddit; ships per-rater labels (human–human κ ceiling) |
| **SemEval-2025 / BRIGHTER** | Cross-lingual transfer target | 32 languages |
| **SemEval-2018 Task 1 E-c** (11, multi-label) | Co-occurrence analysis | Tweets |

A **label-harmonization map** (GoEmotions 27 → Ekman 6 → SemEval labels) is the next
artifact to build before any cross-dataset experiment can run.
