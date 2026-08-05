# Results

Two top-level trees:

```
results/
├── paper_final/          Everything used in the paper
│   ├── data/             CSV/JSON sources  →  see data/README.md (the full file key)
│   └── figures/          The 10 paper figures
└── exploratory/          Archived — NOT used in the paper
    ├── data/             Redundant / superseded runs  →  see data/README.md
    ├── figures/          Archived plots  →  see figures/README.md
    └── server_logs/      Raw GPU-server stdout (provenance)
```

- **Start here for the paper's numbers:** [`paper_final/data/README.md`](paper_final/data/README.md)
  — the file key (naming convention, which file backs which table/figure, column glossaries, recipes).
- **What's archived and why:** [`exploratory/data/README.md`](exploratory/data/README.md)
  and [`exploratory/figures/README.md`](exploratory/figures/README.md).

Everything under `exploratory/` is regenerable from the `paper_final/data/` CSVs via
`scripts/make_figures.py` (standard figures) and `scripts/make_crossdataset_figure.py`
(the cross-dataset figure); it's kept only as a historical record of methods and results.
