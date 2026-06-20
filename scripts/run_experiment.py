"""Run one or more model tiers against a dataset and print a scorecard.

Examples
--------
    # Fast: classical tier on GoEmotions -> Ekman-6, capped to 4k train rows
    python scripts/run_experiment.py --models tfidf_logreg tfidf_linearsvm \
        --dataset go_emotions --limit-train 4000

    # Add the BiLSTM tier
    python scripts/run_experiment.py --models tfidf_logreg bilstm --dataset go_emotions

    # Transformer tier (downloads weights; slow on CPU)
    python scripts/run_experiment.py --models hf_distilbert --dataset go_emotions \
        --limit-train 2000 --limit-test 1000

Datasets are projected to the shared Ekman-6 (+ neutral) schema so every tier
and dataset is scored in the same label space.
"""

from __future__ import annotations

import argparse

from emotion_classification import labels as L
from emotion_classification.experiment import run_experiment
from emotion_classification.loaders import LOADERS
from emotion_classification.preprocessing import prepare_dataset
from emotion_classification.scorecard import Scorecard

# Target label space shared across datasets: Ekman-6 + neutral.
TARGET_LABELS = [*L.EKMAN, L.NEUTRAL]

# Per-dataset native->Ekman projector (see emotion_classification.labels).
PROJECTORS = {
    "go_emotions": L.goemotions_to_ekman,
    "sem_eval_2018_task_1": L.semeval2018_to_ekman,
}


def build_model(name: str):
    """Instantiate a model by short name (lazy imports keep startup cheap)."""
    if name == "tfidf_logreg":
        from emotion_classification.models.classical import TfidfLogReg

        return TfidfLogReg()
    if name == "tfidf_linearsvm":
        from emotion_classification.models.classical import TfidfLinearSVM

        return TfidfLinearSVM()
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
    parser.add_argument("--models", nargs="+", default=["tfidf_logreg"],
                        help="model short names (tfidf_logreg, tfidf_linearsvm, bilstm, hf_distilbert)")
    parser.add_argument("--dataset", default="go_emotions", choices=list(PROJECTORS))
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--limit-train", type=int, default=None,
                        help="cap training rows (speeds up smoke runs)")
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = LOADERS[args.dataset]()
    prepared = prepare_dataset(ds, TARGET_LABELS, PROJECTORS[args.dataset])

    train = prepared[args.train_split]
    test = prepared[args.test_split]
    if args.limit_train:
        train.texts = train.texts[: args.limit_train]
        train.Y = train.Y[: args.limit_train]
    if args.limit_test:
        test.texts = test.texts[: args.limit_test]
        test.Y = test.Y[: args.limit_test]

    print(f"  train={len(train)}  test={len(test)}  labels={TARGET_LABELS}")

    card = Scorecard()
    for name in args.models:
        print(f"\n=== {name} ===")
        model = build_model(name)
        row = run_experiment(model, train, test, dataset_name=args.dataset, seed=args.seed)
        card.add(row)
        print(f"  macro_f1={row.macro_f1:.3f}  micro_f1={row.micro_f1:.3f}  "
              f"ece={row.ece:.3f}  train={row.train_seconds:.1f}s  "
              f"latency={row.predict_latency_ms:.2f}ms  size={row.model_size_mb:.1f}MB")

    print("\n## Raw scorecard\n")
    print(card.to_markdown())
    if len(card) > 1:
        print("\n## Normalized (1 = best per axis)\n")
        norm = card.normalized()
        axes = card.axes()
        header = ["model", *axes]
        print("| " + " | ".join(header) + " |")
        print("| " + " | ".join("---" for _ in header) + " |")
        for r in norm:
            print("| " + " | ".join([r["model"], *[f"{r.get(a, float('nan')):.3f}" for a in axes]]) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
