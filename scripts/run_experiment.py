"""Run one or more model tiers against a dataset and print a scorecard.

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

Label schema (`--schema`) controls the target label space:
    native      dataset-native labels (GoEmotions = 27 emotions + neutral) [default]
    ekman6      collapsed Ekman-6 (+ neutral)
    sentiment3  collapsed positive/negative/ambiguous (+ neutral); GoEmotions only
"""

from __future__ import annotations

import argparse

from emotion_classification import labels as L
from emotion_classification.experiment import run_experiment
from emotion_classification.loaders import LOADERS
from emotion_classification.preprocessing import prepare_dataset
from emotion_classification.results import save_results, save_summary, summarize
from emotion_classification.scorecard import Scorecard

# Classical-tier model short names -> class names in models.classical.
CLASSICAL_BUILDERS = {
    "naive_bayes": "NaiveBayes",
    "logreg": "LogisticReg",
    "linsvm": "LinearSVM",
    "random_forest": "RandomForest",
}

# Dataset -> native->Ekman projector (see emotion_classification.labels).
EKMAN_PROJECTORS = {
    "go_emotions": L.goemotions_to_ekman,
    "sem_eval_2018_task_1": L.semeval2018_to_ekman,
}


def resolve_schema(dataset: str, schema: str, ds):
    """Return ``(label_names, projector)`` for the chosen target schema.

    ``projector=None`` means "use native labels as-is" (the ``native`` schema).
    """
    if schema == "native":
        return list(ds.label_names), None
    if schema == "ekman6":
        if dataset not in EKMAN_PROJECTORS:
            raise SystemExit(f"no Ekman projector for dataset {dataset!r}")
        return [*L.EKMAN, L.NEUTRAL], EKMAN_PROJECTORS[dataset]
    if schema == "sentiment3":
        if dataset != "go_emotions":
            raise SystemExit("sentiment3 schema is only defined for go_emotions")
        return [*L.SENTIMENT, L.NEUTRAL], L.goemotions_to_sentiment
    raise SystemExit(f"unknown schema {schema!r}")


def build_model(name: str, features: str):
    """Instantiate a model by short name (lazy imports keep startup cheap)."""
    if name in CLASSICAL_BUILDERS:
        import emotion_classification.models.classical as C

        return getattr(C, CLASSICAL_BUILDERS[name])(features=features)
    if name == "bilstm":
        from emotion_classification.models.deep import BiLSTMClassifier

        return BiLSTMClassifier()
    if name.startswith("hf_") or name == "transformer":
        from emotion_classification.models.transformer import TransformerClassifier

        model_name = "distilbert-base-uncased" if name in ("hf_distilbert", "transformer") \
            else name[len("hf_"):]
        return TransformerClassifier(model_name=model_name)
    raise SystemExit(f"unknown model: {name!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["logreg"],
                        help="model short names: " + ", ".join(CLASSICAL_BUILDERS) +
                             ", bilstm, hf_<name> (e.g. hf_distilbert)")
    parser.add_argument("--dataset", default="go_emotions", choices=list(LOADERS))
    parser.add_argument("--schema", default="native",
                        choices=["native", "ekman6", "sentiment3"],
                        help="target label space (default: native)")
    parser.add_argument("--features", default="tfidf", choices=["tfidf", "bow"],
                        help="feature backend for classical models (default: tfidf)")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="cap training rows (speeds up smoke runs)")
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="run each model across these seeds and report mean±std "
                             "(overrides --seed; e.g. --seeds 42 43 44 45 46)")
    parser.add_argument("--out-dir", default="results",
                        help="directory for persisted results (default: results/)")
    parser.add_argument("--run-name", default=None,
                        help="basename for saved files (default: <dataset>_<schema>_<features>)")
    parser.add_argument("--no-save", action="store_true", help="skip writing results to disk")
    args = parser.parse_args()

    print(f"Loading {args.dataset} (schema={args.schema}, features={args.features}) ...")
    ds = LOADERS[args.dataset]()
    label_names, projector = resolve_schema(args.dataset, args.schema, ds)
    prepared = prepare_dataset(ds, label_names, projector)

    train = prepared[args.train_split]
    test = prepared[args.test_split]
    if args.limit_train:
        train.texts = train.texts[: args.limit_train]
        train.Y = train.Y[: args.limit_train]
    if args.limit_test:
        test.texts = test.texts[: args.limit_test]
        test.Y = test.Y[: args.limit_test]

    print(f"  train={len(train)}  test={len(test)}  n_labels={len(label_names)}")

    seeds = args.seeds if args.seeds else [args.seed]
    multiseed = len(seeds) > 1

    card = Scorecard()
    for name in args.models:
        print(f"\n=== {name} ===")
        for seed in seeds:
            model = build_model(name, args.features)  # fresh instance per seed
            row = run_experiment(model, train, test, dataset_name=args.dataset, seed=seed,
                                 extra_meta={"schema": args.schema, "features": args.features})
            card.add(row)
            tag = f"[seed {seed}] " if multiseed else ""
            print(f"  {tag}macro_f1={row.macro_f1:.3f}  micro_f1={row.micro_f1:.3f}  "
                  f"ece={row.ece:.3f}  train={row.train_seconds:.1f}s  "
                  f"latency={row.predict_latency_ms:.2f}ms  thrpt={row.throughput:.0f}/s  "
                  f"size={row.model_size_mb:.1f}MB  device={row.device}")

    print("\n## Raw scorecard\n")
    print(card.to_markdown())

    if multiseed:
        print(f"\n## Multi-seed summary (mean +/- std over {len(seeds)} seeds)\n")
        for s in summarize(card):
            print(f"  {s['model']}:  "
                  f"macro_f1={s['macro_f1_mean']:.3f}+/-{s['macro_f1_std']:.3f}  "
                  f"micro_f1={s['micro_f1_mean']:.3f}+/-{s['micro_f1_std']:.3f}  "
                  f"train={s['train_seconds_mean']:.1f}+/-{s['train_seconds_std']:.1f}s  "
                  f"size={s['model_size_mb_mean']:.1f}MB")
    elif len(card) > 1:
        print("\n## Normalized (1 = best per axis)\n")
        norm = card.normalized()
        axes = card.axes()
        header = ["model", *axes]
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join("---" for _ in header) + " |")
        for r in norm:
            print("| " + " | ".join([r["model"], *[f"{r.get(a, float('nan')):.3f}" for a in axes]]) + " |")

    if not args.no_save:
        run_name = args.run_name or f"{args.dataset}_{args.schema}_{args.features}"
        paths = save_results(card, args.out_dir, run_name)
        if multiseed:
            paths["summary"] = save_summary(card, args.out_dir, run_name)
        print("\nSaved results:")
        for kind, path in paths.items():
            print(f"  {kind:<11} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
