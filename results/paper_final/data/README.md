# Results — File Key

Navigation guide to `results/`, which splits into two top-level trees:

```
results/
├── paper_final/          ← everything used in the paper
│   ├── data/             (the CSV/JSON sources — YOU ARE HERE)
│   └── figures/          (the 10 paper figures)
└── exploratory/          ← archived, NOT used in the paper
    ├── data/             (redundant / superseded runs)
    ├── figures/          (archived plots)
    └── server_logs/      (raw GPU-server stdout, provenance)
```

Every experiment writes a small family of files that share a **stem**; figures are named
from the same stems. Once you can read a stem you can find any number in the paper. Paths
below are relative to `results/`. This key documents the **`paper_final/data/`** files
(where it lives); the archived runs are catalogued in `exploratory/data/README.md` and
`exploratory/figures/README.md`. The tokens `all` / `bow` / `crosstier` and the
per-schema 3-epoch transformer stems appear below for completeness but their files live
under `exploratory/data/`, not here.

---

## 1. How a filename is built

```
go_native_final_scorecard.csv
│  │      │      └── artifact type  (what kind of file)
│  │      └───────── run group      (which experiment)
│  └──────────────── label schema   (how many emotion classes)
└─────────────────── dataset        (go = GoEmotions)
```

Cross-dataset files use an arrow-style stem instead: `xdata_go2semeval` = trained on
GoEmotions, tested on SemEval-2018 (and `xdata_semeval2go` is the reverse).

### 1a. Dataset prefix
| Token | Meaning |
|---|---|
| `go_` | GoEmotions (primary corpus, Reddit, 43,410 train / 5,427 test) |
| `sem_` | In-domain SemEval-2018 (tweets, 6,838 train / 3,259 test); e.g. `sem_ekman6` = train+test on SemEval — the reverse-transfer baseline |
| `xdata_go2semeval` | Cross-dataset transfer: train GoEmotions → test SemEval-2018, shared Ekman-6 space |
| `xdata_semeval2go` | **Reverse** transfer: train SemEval-2018 → test GoEmotions, shared Ekman-6 |
| `gpu_smoke` | Tiny GPU sanity-check run — **not a result**, ignore for the paper |

### 1b. Label schema
| Token | Classes | Notes |
|---|---|---|
| `native` | 27 emotions + neutral | **Primary** schema for all headline results |
| `ekman6` | 6 Ekman emotions + neutral | Coarser; also the shared space for cross-dataset |
| `sentiment3` | positive / negative / ambiguous + neutral | Coarsest |
| `granularity` | — | Not a schema — a *comparison across* all three (see §3) |

### 1c. Run group (the experiment)
| Token | What it is |
|---|---|
| `all` | The four **classical** models, TF-IDF features — *archived → `exploratory/data/`* |
| `bow` | The four classical models, **Bag-of-Words** features — *archived → `exploratory/data/`* |
| `features` | TF-IDF **vs** BoW side by side (8 classical rows) → feature comparison |
| `bert` / `distilbert` / `bilstm` | A single model run *(bare per-schema 3-epoch bert/distilbert are archived; native 3-epoch and all `_ep5` stay here)* |
| `final` | The **headline** seven-model table + figures for the paper (5-epoch transformers) |
| `crosstier` | Earlier seven-model table with **3-epoch** transformers — **superseded by `final`**; *archived → `exploratory/data/`*, don't cite |
| `reduced` | Matched-size control: GoEmotions train **subsampled to 6,838** (= SemEval size); see §3 |
| `lw_server` | Lightweight (classical) tier re-run on the GPU server |
| `seeds` | **Multi-seed** run (3 seeds) → adds mean±std for stochastic models |
| `ep5` / `ep8` | Transformer trained for **5 / 8 epochs** → training-budget analysis |
| `transformers_ep5_seeds` | Both transformers, 5 epochs, multi-seed (the fair, final transformer numbers) |

**Modifiers** that can attach to a stem: `_ep5` (5-epoch transformers — the values to
cite), and `_s43` / `_s44` (subsample seed for the `reduced` control; the unsuffixed
`reduced` run is seed 42). Worked example — `xdata_go2semeval_reduced_s43_ep5` reads as:
forward transfer (go→semeval) · matched-size subsample · seed 43 · 5-epoch.

### 1d. Artifact type (suffix)
| Suffix | Contents |
|---|---|
| `_scorecard.csv` | **The deployment scorecard** — one row per model: accuracy + efficiency headline metrics (see §4a) |
| `_per_emotion.csv` | Per-emotion F1 + bootstrap CI + support, one row per (model, emotion) (see §4b) |
| `.json` | Full raw results the CSVs are derived from (config, all metrics, run metadata) |
| `_seeds*.` | The multi-seed variant of the above (per-seed rows) |
| `_seeds_summary.csv` | Multi-seed **summary** — mean & std of every metric across seeds (see §4c) |

---

## 2. Where each paper number lives

| Paper element | File(s) |
|---|---|
| **Table 1** — main scorecard (7 models, native-27) | `go_native_final_scorecard.csv` (headline; 5-epoch) |
| **Table 2** — training-budget / epochs | `go_native_bert_ep5/ep8_scorecard.csv`, `go_native_distilbert_ep5/ep8_scorecard.csv`, `go_native_transformers_ep5_seeds_summary.csv` |
| **Cross-dataset (§4.3/4.7)** — the corpus-specificity result, three regimes | **Full:** `xdata_go2semeval_per_emotion.csv` (classical) + `xdata_go2semeval_ep5_*` (transformers). **Matched-size:** `xdata_go2semeval_reduced_ep5_*` + `_s43` + `_s44`, with in-domain `go_ekman6_reduced_ep5_*`. **Reverse:** `xdata_semeval2go_ep5_*` + in-domain `sem_ekman6_ep5_*` |
| **Per-emotion F1 + CIs (rare emotions)** | `go_native_final_per_emotion.csv` (widths grow as `support` falls) |
| **Granularity sweep (27 → 6 → 3)** | `go_granularity_scorecard.csv` (classical); transformer 5-epoch points from `go_{native,ekman6,sentiment3}_{bert,distilbert}_ep5_scorecard.csv` |
| **Feature comparison (TF-IDF vs BoW)** | `go_native_features_scorecard.csv` |
| **Uncertainty (multi-seed)** | any `*_seeds_summary.csv` (mean±std) |

> **Cross-dataset metric:** all transfer numbers are macro-F1 over the **6 shared Ekman
> emotions, neutral excluded** (SemEval has no neutral). Lead with *absolute* cross-domain
> F1, not retention (retention flatters the lightweight tier). The headline is that
> transfer is **corpus/direction-specific, not a size effect** — see the paper's §4.3.

> **Which transformer numbers are the "real" ones?** Use the **5-epoch** results
> (`*_ep5_*`, `*_transformers_ep5_seeds_*`). The bare `go_native_bert`/`distilbert`
> files are the earlier 3-epoch runs that under-trained the transformers — kept only
> as the "before" side of the training-budget story, not for headline claims.

---

## 3. Granularity files (special case)

`go_granularity` stacks the **same classical models across all three schemas** into one
table so the coarsening effect is visible in a single file (hence the repeated model rows
— the `schema` column tells them apart: native / ekman6 / sentiment3). The BoW version
(`go_granularity_bow`) is archived under `exploratory/data/`.

---

## 4. Column glossaries

### 4a. `*_scorecard.csv`
| Column | Description |
|---|---|
| `model` | Model identifier (e.g. `logreg_tfidf`, `hf_bert-base-uncased`) |
| `dataset` | Source dataset (`go_emotions`) |
| `schema` | Label schema (native / ekman6 / sentiment3) |
| `features` | Text representation (`tfidf` / `bow`) |
| `device` | Run hardware (`cpu` / `cuda`) — timing is only comparable within a device class |
| `macro_f1` | Macro-F1 (unweighted mean of per-label F1) |
| `macro_f1_low` / `macro_f1_high` | 95% bootstrap CI (2.5th / 97.5th percentile) for macro-F1 |
| `micro_f1` | Micro-F1 (pooled label decisions) |
| `micro_f1_low` / `micro_f1_high` | 95% bootstrap CI for micro-F1 |
| `subset_accuracy` | Exact-match accuracy (strictest multi-label metric) |
| `ece` | Expected Calibration Error |
| `train_seconds` | Wall-clock training time |
| `predict_latency_ms` | Mean inference time per example (ms) |
| `throughput` | Examples classified per second |
| `model_size_mb` | On-disk footprint (hardware-independent) |
| `cost_usd` | Estimated run cost (optional; blank where not recorded) |
| `n_train` / `n_test` | Training / test example counts |
| `n_labels` | Number of labels in the schema |
| `seed` | Random seed for this run |

### 4b. `*_per_emotion.csv`
| Column | Description |
|---|---|
| `model` | Model identifier |
| `dataset` | Source dataset |
| `schema` | Label schema |
| `seed` | Random seed for this run |
| `emotion` | The emotion label this row scores |
| `f1` | Per-emotion F1 |
| `f1_low` / `f1_high` | 95% bootstrap CI for that F1 — widens sharply as `support` falls (rare-emotion imprecision, e.g. *grief*, 6 examples) |
| `support` | Number of positive test examples for that emotion |

### 4c. `*_seeds_summary.csv`
Source for any "±" reported in the paper.

| Column | Description |
|---|---|
| `model` | Model identifier |
| `dataset` | Source dataset |
| `schema` | Label schema |
| `features` | Text representation |
| `n_seeds` | Number of seeds averaged (3) |
| `<metric>_mean` | Mean across seeds |
| `<metric>_std` | Standard deviation across seeds |

`<metric>` covers `macro_f1`, `micro_f1`, `subset_accuracy`, `ece`, `train_seconds`,
`predict_latency_ms`, `throughput`, and `model_size_mb` (each with its own `_mean` and
`_std` column).

---

## 5. Figures (`paper_final/figures/`)

The paper's 10 figures. Named `go_<schema>_<group>_<plottype>.png`; the **plot type**
is the last token:

| Plot type | Shows | Paper role |
|---|---|---|
| `pareto_size` / `pareto_train` | trade-off vs **model size** / vs **training time** | the efficiency-cost story |
| `heatmap` | per-emotion F1 grid (models × emotions) | per-emotion behaviour |
| `violin` | distribution of per-emotion F1 per model | spread across emotions |
| `f1_vs_frequency` | per-emotion F1 vs support (with CIs) | rare-emotion imprecision |
| `macro_f1` / `micro_f1` | line plots across conditions | granularity & feature sweeps |

**The headline figures** are the `go_native_final_*` set (pareto_size, pareto_train,
heatmap, violin, f1_vs_frequency) — 5-epoch, all seven models — plus the two
`go_granularity_*` and two `go_native_features_*` bar charts.

**`cross_dataset_transfer.png`** (special, not a `go_*` stem) — the corpus-specificity
result: absolute cross-domain F1 for LogReg/DistilBERT/BERT across the three regimes
(Full / Matched-size±std / Reverse). Generated by `scripts/make_crossdataset_figure.py`,
not `make_figures.py`.

> The classical-only `_all` / `_bow` plots and the 3-epoch `crosstier` figures are **not**
> here — they're archived under `exploratory/figures/` (see its README).

---

## 6. Logs (`exploratory/server_logs/`)

Raw stdout from each GPU-server run, one `.log` per run stem (plus `run_m3_full.log`,
the full M3 batch). Not needed to read results — kept for provenance, timing sanity,
and debugging. Moved under `exploratory/` since they're provenance, not paper output.

---

### Quick recipes
- *"The main results table?"* → `go_native_final_scorecard.csv`
- *"Did the transformer really only win by 0.045 macro-F1?"* → compare `logreg_tfidf`
  vs `hf_bert-base-uncased` rows in `go_native_final_scorecard.csv`
- *"The cross-dataset result?"* → run `python scripts/make_crossdataset_figure.py` (prints
  the numbers + writes the figure); or read the three regimes per §2. Headline: LogReg
  leads from GoEmotions at full **and** matched size and is far more stable; the ordering
  flips only from the SemEval source — corpus-specific, not size.
- *"Is the matched-size flip real?"* → it isn't; compare `xdata_go2semeval_reduced_ep5`,
  `_s43`, `_s44` — BERT's cross-F1 swings 0.34/0.30/0.24 while LogReg holds ~0.31.
- *"Error bars on grief?"* → `go_native_final_per_emotion.csv`, row emotion=`grief`
- *"3-epoch vs 5-epoch gap?"* → `go_native_bert_ep5_scorecard.csv` vs the 3-epoch
  `go_native_bert_scorecard.csv`
