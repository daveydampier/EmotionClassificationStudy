r"""Run one or more models against a corpus and print a scorecard.

Each model is trained once per seed and then predicts on every evaluation set, so
one run can score the in-domain test split and any number of other corpora
without retraining. The per-example probabilities are cached next to the results.

Examples
--------
    # Lightweight tier on GoEmotions, full native 27-class taxonomy (the default)
    python scripts/run_experiment.py \
        --models naive_bayes logreg linsvm random_forest \
        --dataset go_emotions --limit-train 4000 --limit-test 1500

    # Bag-of-Words features instead of TF-IDF
    python scripts/run_experiment.py --models logreg --features bow

    # Add the BiLSTM tier and the transformer reference
    python scripts/run_experiment.py --models logreg bilstm hf_distilbert

    # Granularity sweep: same models under a coarser schema
    python scripts/run_experiment.py --models logreg --schema ekman6

    # One fit, two evaluation sets: in-domain test plus cross-corpus transfer
    python scripts/run_experiment.py --models logreg --schema ekman6 \
        --eval-sets go_emotions:test sem_eval_2018_task_1:test

Label schema (`--schema`) controls the target label space:
    native      corpus-native labels (GoEmotions = 27 emotions + neutral) [default]
    ekman6      collapsed Ekman-6 (+ neutral)
    sentiment3  collapsed positive/negative/ambiguous (+ neutral); GoEmotions only

Evaluation sets (`--eval-sets`) are named `<corpus>:<split>`. The default is the
training corpus's test split. The training corpus's validation split is also
predicted and cached (never scored) so thresholds can be tuned later; turn that
off with `--val-split none`. Predictions go to `<out-dir>/predictions/<run-name>/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from emotion_classification.corpora import SCHEMAS, corpus_names
from emotion_classification.errors import ConfigError
from emotion_classification.models.registry import model_names
from emotion_classification.pipeline import RunConfig, run_pipeline
from emotion_classification.results import save_results, save_summary, summarize
from emotion_classification.scorecard import Scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["logreg"],
                        help="model short names: " + ", ".join(model_names()) +
                             ", or hf_<checkpoint> (e.g. hf_distilbert)")
    parser.add_argument("--dataset", default="go_emotions", choices=corpus_names(),
                        help="training corpus (default: go_emotions)")
    parser.add_argument("--schema", default="native", choices=list(SCHEMAS),
                        help="target label space (default: native)")
    parser.add_argument("--features", default="tfidf", choices=["tfidf", "bow"],
                        help="feature backend for classical models (default: tfidf)")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--eval-sets", nargs="+", default=None, metavar="CORPUS:SPLIT",
                        help="evaluation sets to score; each model is trained once and "
                             "predicts on all of them (default: <dataset>:<test-split>)")
    parser.add_argument("--test-dataset", default=None, choices=corpus_names(),
                        help="shorthand for --eval-sets <corpus>:<test-split> (cross-dataset "
                             "generalization). Requires a shared schema, e.g. ekman6.")
    parser.add_argument("--val-split", default="validation",
                        help="training-corpus split whose predictions are cached, unscored, "
                             "for later tuning (default: validation; 'none' turns it off)")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="cap training rows (speeds up smoke runs)")
    parser.add_argument("--limit-test", "--limit-eval", dest="limit_eval", type=int,
                        default=None, help="cap the rows of every evaluation set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="run each model across these seeds and report mean±std "
                             "(overrides --seed; e.g. --seeds 42 43 44 45 46)")
    parser.add_argument("--out-dir", default="results",
                        help="directory for persisted results (default: results/; thesis "
                             "batches pass results/thesis/<batch>)")
    parser.add_argument("--run-name", default=None,
                        help="basename for saved files (default: <dataset>_<schema>_<features>)")
    parser.add_argument("--no-save", action="store_true",
                        help="skip writing results and predictions to disk")
    parser.add_argument("--no-predictions", action="store_true",
                        help="write the scorecard files but not the prediction cache")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="bootstrap resamples for test-set F1 confidence intervals "
                             "(0 = off; e.g. 1000). Adds macro/micro + per-emotion CIs.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="training epochs for deep/transformer tiers (overrides the "
                             "model default; e.g. 5). Ignored by classical models.")
    parser.add_argument("--subsample-train", type=int, default=None,
                        help="randomly sample this many training rows before fitting, for "
                             "matched-size control experiments. Each seed draws its own "
                             "subsample. Unlike --limit-train (first N), this is random.")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Translate parsed arguments into a :class:`RunConfig`."""
    if args.eval_sets and args.test_dataset:
        raise SystemExit("use either --eval-sets or --test-dataset, not both")
    eval_sets = args.eval_sets
    if args.test_dataset:
        eval_sets = [f"{args.test_dataset}:{args.test_split}"]
    save_predictions = not (args.no_save or args.no_predictions)
    return RunConfig(
        models=list(args.models),
        dataset=args.dataset,
        schema=args.schema,
        features=args.features,
        train_split=args.train_split,
        test_split=args.test_split,
        eval_sets=eval_sets,
        val_split=None if args.val_split.lower() == "none" else args.val_split,
        seeds=list(args.seeds) if args.seeds else [args.seed],
        epochs=args.epochs,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        subsample_train=args.subsample_train,
        bootstrap=args.bootstrap,
        predictions_dir=Path(args.out_dir) / "predictions" if save_predictions else None,
        run_name=args.run_name or f"{args.dataset}_{args.schema}_{args.features}",
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    print(f"Loading {cfg.dataset} (schema={cfg.schema}, features={cfg.features}) ...")
    try:
        result = run_pipeline(cfg, log=print)
    except ConfigError as err:
        raise SystemExit(str(err)) from None
    card = result.card
    multiseed = len(cfg.seeds) > 1

    print("\n## Raw scorecard\n")
    print(card.to_markdown())

    if multiseed:
        print(f"\n## Multi-seed summary (mean +/- std over {len(cfg.seeds)} seeds)\n")
        for s in summarize(card):
            print(f"  {s['model']} [{s['dataset']}]:  "
                  f"macro_f1={s['macro_f1_mean']:.3f}+/-{s['macro_f1_std']:.3f}  "
                  f"micro_f1={s['micro_f1_mean']:.3f}+/-{s['micro_f1_std']:.3f}  "
                  f"train={s['train_seconds_mean']:.1f}+/-{s['train_seconds_std']:.1f}s  "
                  f"size={s['model_size_mb_mean']:.1f}MB")
    else:
        # Models are only comparable on the same evaluation set, so normalize
        # within each one rather than across the whole card.
        for dataset in dict.fromkeys(row.dataset for row in card.rows):
            group = Scorecard([row for row in card.rows if row.dataset == dataset])
            if len(group) < 2:
                continue
            print(f"\n## Normalized on {dataset} (1 = best per axis)\n")
            axes = group.axes()
            header = ["model", *axes]
            print("| " + " | ".join(header) + " |")
            print("| " + " | ".join("---" for _ in header) + " |")
            for r in group.normalized():
                cells = [r["model"], *[f"{r.get(a, float('nan')):.3f}" for a in axes]]
                print("| " + " | ".join(cells) + " |")

    if not args.no_save:
        paths = save_results(card, args.out_dir, cfg.run_name)
        if multiseed:
            paths["summary"] = save_summary(card, args.out_dir, cfg.run_name)
        print("\nSaved results:")
        for kind, path in paths.items():
            print(f"  {kind:<11} {path}")
        if result.prediction_paths:
            print(f"  {'predictions':<11} {len(result.prediction_paths)} files in "
                  f"{result.prediction_paths[0].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
