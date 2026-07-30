# Results — File Key

A navigation guide to everything in `results/`. Every experiment writes a small
family of files that share a **stem**; figures are named from the same stems. Once
you can read a stem, you can find any number in the paper.

- **`*.json` / `*.csv`** — the numbers (this folder).
- **`figures/*.png`** — the plots.
- **`server_logs/*.log`** — raw stdout from each GPU-server run (provenance/debugging).

---

## 1. How a filename is built

```
go_native_crosstier_scorecard.csv
│  │      │         └── artifact type  (what kind of file)
│  │      └────────────── run group     (which experiment)
│  └───────────────────── label schema  (how many emotion classes)
└──────────────────────── dataset       (go = GoEmotions)
```

Cross-dataset files use an arrow-style stem instead: `xdata_go2semeval` = trained on
GoEmotions, tested on SemEval-2018.

### 1a. Dataset prefix
| Token | Meaning |
|---|---|
| `go_` | GoEmotions (primary corpus, Reddit, 43,410 train / 5,427 test) |
| `xdata_go2semeval` | Cross-dataset transfer: train GoEmotions → test SemEval-2018 (tweets), shared Ekman-6 space |
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
| `all` | The four **classical** models, TF-IDF features |
| `bow` | The four classical models, **Bag-of-Words** features |
| `features` | TF-IDF **vs** BoW side by side (8 classical rows) → feature comparison |
| `bert` / `distilbert` / `bilstm` | A single model run |
| `crosstier` | All **seven** models in one table (classical + BiLSTM + both transformers) |
| `final` | The **headline** seven-model table + figures for the paper |
| `lw_server` | Lightweight (classical) tier re-run on the GPU server |
| `seeds` | **Multi-seed** run (3 seeds) → adds mean±std for stochastic models |
| `ep5` / `ep8` | Transformer trained for **5 / 8 epochs** → training-budget analysis |
| `transformers_ep5_seeds` | Both transformers, 5 epochs, multi-seed (the fair, final transformer numbers) |

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
| **Table 1** — main scorecard (7 models, native-27) | `go_native_final_scorecard.csv` (headline), `go_native_crosstier_scorecard.csv` |
| **Table 2** — training-budget / epochs | `go_native_bert_ep5/ep8_scorecard.csv`, `go_native_distilbert_ep5/ep8_scorecard.csv`, `go_native_transformers_ep5_seeds_summary.csv` |
| **Table 3** — cross-dataset transfer | `xdata_go2semeval_scorecard.csv`, `xdata_go2semeval_per_emotion.csv` |
| **§4.4** — per-emotion F1 + CIs (rare emotions) | `go_native_final_per_emotion.csv` (widths grow as `support` falls) |
| **§4.5** — granularity sweep (27 → 6 → 3) | `go_granularity_scorecard.csv`, `go_granularity_bow_scorecard.csv` |
| **§4.6** — feature comparison (TF-IDF vs BoW) | `go_native_features_scorecard.csv` |
| **Uncertainty (multi-seed)** | any `*_seeds_summary.csv` (mean±std) |

> **Which transformer numbers are the "real" ones?** Use the **5-epoch** results
> (`*_ep5_*`, `*_transformers_ep5_seeds_*`). The bare `go_native_bert`/`distilbert`
> files are the earlier 3-epoch runs that under-trained the transformers — kept only
> as the "before" side of the training-budget story, not for headline claims.

---

## 3. Granularity files (special case)

`go_granularity*` stack the **same models across all three schemas** into one table so
the coarsening effect is visible in a single file (hence the repeated model rows — the
`schema` column tells them apart: native / ekman6 / sentiment3). `_bow` is the same
comparison on Bag-of-Words features.

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

## 5. Figures (`figures/`)

Named `go_<schema>_<group>_<plottype>.png`. Same schema/group tokens as above; the
**plot type** is the last token:

| Plot type | Shows | Paper role |
|---|---|---|
| `pareto` | efficiency ↔ accuracy trade-off frontier | the efficiency-cost story |
| `pareto_size` / `pareto_train` | trade-off vs **model size** / vs **training time** (split view) | native cross-tier & final |
| `heatmap` | per-emotion F1 grid (models × emotions) | per-emotion behaviour |
| `violin` | distribution of per-emotion F1 per model | spread across emotions |
| `f1_vs_frequency` | per-emotion F1 vs support (with CIs) | rare-emotion imprecision (§4.4) |
| `macro_f1` / `micro_f1` | line plots across conditions | granularity & feature sweeps |

**The paper's headline figures** are the `go_native_final_*` set (pareto_size,
pareto_train, heatmap, violin, f1_vs_frequency). The `go_native_crosstier_*` set is
the same view from the full 7-model comparison. `_all` / `_bow` figures are the
classical-only and BoW variants; `*_features_*` and `*_granularity_*` back §4.6 and
§4.5.

---

## 6. `server_logs/`

Raw stdout from each GPU-server run, one `.log` per run stem (plus `run_m3_full.log`,
the full M3 batch). Not needed to read results — kept for provenance, timing sanity,
and debugging. `run_m3_full.log` is the umbrella log for the final batch.

---

### Quick recipes
- *"The main results table?"* → `go_native_final_scorecard.csv`
- *"Did the transformer really only win by 0.045 macro-F1?"* → compare `logreg_tfidf`
  vs `hf_bert-base-uncased` rows in `go_native_final_scorecard.csv`
- *"The cross-dataset headline (65% vs 36–43%)?"* → `xdata_go2semeval_scorecard.csv`
- *"Error bars on grief?"* → `go_native_final_per_emotion.csv`, row emotion=`grief`
- *"3-epoch vs 5-epoch gap?"* → `go_native_bert_ep5_scorecard.csv` vs the 3-epoch
  `go_native_bert_scorecard.csv`
