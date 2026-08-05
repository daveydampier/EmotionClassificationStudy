# Exploratory / superseded results (not used in the paper)

Historical record. None of these files are a source for any table, figure, or number in
the manuscript — the paper's sources all live one level up in `results/`. Kept for
provenance; all are regenerable via `scripts/run_experiment.py`.

## What's here, and why it's not in the paper

**A. Redundant classical breakdowns** — single-schema / single-feature classical runs
that were *aggregated* into the files the paper actually uses:
- `go_native_all`, `go_ekman6_all`, `go_sentiment3_all` — classical (TF-IDF) per schema;
  the schema sweep is reported from `go_granularity`, and native efficiency from
  `go_native_lw_server`.
- `go_native_bow`, `go_ekman6_bow`, `go_sentiment3_bow` — classical (BoW) per schema;
  the TF-IDF-vs-BoW comparison is reported from `go_native_features`.
- `go_granularity_bow` — BoW granularity sweep; the paper uses the TF-IDF version
  (`go_granularity`).

**B. Superseded 3-epoch per-schema transformers** — replaced by their `*_ep5` versions
once every transformer number was standardized to 5 epochs (the §4.5 granularity table
and the cross-dataset denominators use the `_ep5` runs):
- `go_ekman6_bert`, `go_ekman6_distilbert`, `go_sentiment3_bert`, `go_sentiment3_distilbert`
  (and their `_seeds` multi-seed variants).
- *(The native 3-epoch runs — `go_native_bert`, `go_native_distilbert`, `*_seeds` — are
  NOT here: they stay in `results/` as the "before" side of the epochs analysis, §4.3.)*

**C. Non-paper runs:**
- `gpu_smoke` — a tiny GPU sanity-check, never a result.
- `go_native_crosstier` — an earlier 3-epoch, 7-model assembly, superseded by
  `go_native_final` (5-epoch). Its figures were removed; the CSVs are kept here.

Each stem has `.json` + `_scorecard.csv` + `_per_emotion.csv` (multi-seed stems also
`_summary.csv`).
