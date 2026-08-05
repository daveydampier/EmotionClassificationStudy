# Exploratory figures (not used in the paper)

Historical record from the M1 exploration phase. These are **classical-only,
single-schema/single-feature breakdowns** — superseded for the paper by the 7-model
`go_native_final_*` set (efficiency + per-emotion) and by the summary bar charts
(`go_granularity_*`, `go_native_features_*`) in the parent `figures/` directory.

Kept for provenance; **not cited in the manuscript**. All are regenerable from the CSVs
in `results/` via `scripts/make_figures.py --name <stem>`.

Contents:
- `go_native_all_*`, `go_native_bow_*` — 4-model classical, native-27 (TF-IDF / BoW)
- `go_ekman6_all_*`, `go_ekman6_bow_*` — classical, Ekman-6
- `go_sentiment3_all_*`, `go_sentiment3_bow_*` — classical, sentiment-3
- `go_granularity_bow_*` — BoW granularity sweep (the paper uses the TF-IDF version)

The paper's figures live one level up in `results/figures/`.
