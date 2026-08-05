"""Generate the study's figures from persisted scorecard results.

Reads the CSVs written by ``run_experiment.py`` (``<name>_scorecard.csv`` and
``<name>_per_emotion.csv``) and writes PNGs — no models are re-run.

Defaults read from ``results/paper_final/data`` and write to
``results/paper_final/figures`` (override with ``--results-dir`` / ``--out-dir``).

Example
-------
    python scripts/make_figures.py --name go_native_final
    #   -> results/paper_final/figures/go_native_final_{pareto_train,pareto_size,
    #      heatmap,violin,f1_vs_frequency}.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_classification import viz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results/paper_final/data",
                        help="directory holding the persisted CSVs "
                             "(default: results/paper_final/data)")
    parser.add_argument("--name", required=True,
                        help="run basename, e.g. go_native_final")
    parser.add_argument("--out-dir", default="results/paper_final/figures",
                        help="where to write PNGs (default: results/paper_final/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    scorecard = viz.load_scorecard(results_dir / f"{args.name}_scorecard.csv")
    per_emotion = viz.load_per_emotion(results_dir / f"{args.name}_per_emotion.csv")

    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    paths = viz.save_all(scorecard, per_emotion, out_dir, prefix=f"{args.name}_")

    print(f"Wrote {len(paths)} figures to {out_dir}:")
    for name, path in paths.items():
        print(f"  {name:<16} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
