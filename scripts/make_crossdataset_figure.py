"""Cross-dataset transfer figure — the corpus-specificity result.

Grouped bars of *absolute* cross-domain macro-F1 (over the 6 shared Ekman
emotions, neutral excluded) for the lightweight baseline and the two
transformers, across three transfer regimes:

    Full      : train full GoEmotions (43k) -> test SemEval
    Matched   : train GoEmotions subsampled to 6,838 -> test SemEval
                (mean +/- std over 3 random draws; error bars)
    Reverse   : train SemEval (6.8k) -> test GoEmotions

The story it shows: LogReg leads from GoEmotions at full *and* matched size and
is far more stable (tiny error bar); the ordering flips only when the source
corpus changes (Reverse). So transfer is corpus/direction-specific, not a size
effect. Run after the M4/M5/M6 results are in ``results/``.

    python scripts/make_crossdataset_figure.py
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

EKMAN6 = {"anger", "disgust", "fear", "joy", "sadness", "surprise"}
MODELS = [
    ("logreg_tfidf", "LogReg"),
    ("hf_distilbert-base-uncased", "DistilBERT"),
    ("hf_bert-base-uncased", "BERT"),
]


def macro6(path: Path, model: str) -> float:
    """Mean F1 over the 6 shared Ekman emotions for one model.

    Each cross-dataset/reduced results file is a single seed, so no seed filter is
    needed (the reduced draws live in separate _s43/_s44 files).
    """
    f1s: dict[str, float] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["model"] == model and row["emotion"] in EKMAN6:
                f1s[row["emotion"]] = float(row["f1"])
    if not f1s:
        raise SystemExit(f"no {model!r} rows found in {path}")
    return sum(f1s.values()) / len(f1s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rd = Path(args.results_dir)

    import matplotlib.pyplot as plt
    import numpy as np

    # --- gather absolute cross-domain macro-F1 per regime -------------------
    # Full: LogReg from the classical xdata run, transformers from the ep5 run.
    full = {}
    for key, _ in MODELS:
        src = "xdata_go2semeval_ep5" if key.startswith("hf_") else "xdata_go2semeval"
        full[key] = (macro6(rd / f"{src}_per_emotion.csv", key), 0.0)
    # Matched: 3 draws -> mean, std.
    matched = {}
    for key, _ in MODELS:
        vals = [macro6(rd / f"xdata_go2semeval_reduced{sfx}_ep5_per_emotion.csv", key)
                for sfx in ("", "_s43", "_s44")]
        matched[key] = (st.mean(vals), st.pstdev(vals))
    # Reverse: single run.
    reverse = {key: (macro6(rd / "xdata_semeval2go_ep5_per_emotion.csv", key), 0.0)
               for key, _ in MODELS}

    regimes = [
        ("Full\nGoEmotions(43k)→SemEval", full),
        ("Matched size\nGoEmotions(6.8k)→SemEval", matched),
        ("Reverse\nSemEval(6.8k)→GoEmotions", reverse),
    ]

    # --- plot ----------------------------------------------------------------
    colors = {"logreg_tfidf": "#2E7D32", "hf_distilbert-base-uncased": "#5B8FF9",
              "hf_bert-base-uncased": "#1A3B8B"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    n_models = len(MODELS)
    width = 0.26
    x = np.arange(len(regimes))
    for i, (key, label) in enumerate(MODELS):
        means = [reg[key][0] for _, reg in regimes]
        errs = [reg[key][1] for _, reg in regimes]
        bars = ax.bar(x + (i - (n_models - 1) / 2) * width, means, width,
                      yerr=errs, capsize=4, label=label, color=colors[key],
                      error_kw={"elinewidth": 1.3})
        for b, m in zip(bars, means):
            ax.annotate(f"{m:.3f}", (b.get_x() + b.get_width() / 2, m),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in regimes], fontsize=9)
    ax.set_ylabel("absolute cross-domain macro-F1\n(6 shared Ekman emotions)")
    ax.set_ylim(0, 0.47)
    ax.set_title("Cross-dataset transfer is corpus-specific, not a size effect\n"
                 "LogReg leads from GoEmotions at full and matched size and is the most "
                 "stable;\nthe ordering flips only when the source corpus changes",
                 fontsize=10.5)
    ax.legend(title="model", loc="upper center", ncol=3, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = Path(args.out) if args.out else rd / "figures" / "cross_dataset_transfer.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")
    # echo the numbers for the numbers-audit / caption
    for name, reg in regimes:
        line = "  ".join(f"{lbl} {reg[k][0]:.3f}±{reg[k][1]:.3f}" for k, lbl in MODELS)
        print(f"  {name.splitlines()[0]:12} {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
