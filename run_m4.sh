#!/usr/bin/env bash
# M4 GPU batch: epoch-consistency polish + reverse-direction cross-dataset.
#
# Goal: make EVERY transformer number 5-epoch (closing the "under-trained
# transformers" objection) and add the reverse transfer (SemEval -> GoEmotions)
# so the robustness finding is BIDIRECTIONAL, not one-way.
#
# Maps to REPORT_NOTES section 7 optional items:
#   Item 3  granularity transformers @5ep  (ekman6 + sentiment3)
#           -> these ekman6 runs also serve as the FORWARD in-domain denominators
#   Item 1  forward cross-dataset transformers @5ep  (GoEmotions -> SemEval)
#   Item 2  reverse-direction transfer @5ep  (SemEval -> GoEmotions)
#           + its in-domain SemEval-ekman6 baseline (the reverse denominators)
#
# Classical models ignore --epochs (they don't train by epochs); BiLSTM is left
# at its existing runs (the 3->5 epoch issue was transformer-specific).
# New runs are suffixed _ep5 so nothing existing is overwritten.
set -u
cd "$(dirname "$0")"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1          # pick an idle GPU; adjust if 1 is busy
mkdir -p logs

run () {   # run <run-name> <args...>
  echo ">>> $1  ($(date +%H:%M:%S))"
  if python scripts/run_experiment.py "${@:2}" --run-name "$1" > "logs/$1.log" 2>&1; then
    grep -E "macro_f1=" "logs/$1.log" | sed "s/^/    /"
  else
    echo "    !! FAILED — tail of logs/$1.log:"; tail -n 6 "logs/$1.log" | sed "s/^/    /"
  fi
}

# --- Preflight: does SemEval-2018 have a train split? (Item 2 needs it) -------
echo "== Preflight: SemEval-2018 splits =="
SEMEVAL_OK=$(python - <<'PY'
try:
    from emotion_classification.loaders import load_semeval2018
    ds = load_semeval2018()
    sizes = {k: len(v) for k, v in ds.splits.items()}
    print("SIZES", sizes)
    print("yes" if ("train" in ds.splits and "test" in ds.splits) else "no")
except Exception as e:
    print("ERROR", repr(e))
    print("no")
PY
)
echo "$SEMEVAL_OK" | sed "s/^/    /"
SEMEVAL_OK=$(printf '%s\n' "$SEMEVAL_OK" | tail -n 1)

echo
echo "############################################################"
echo "# Item 3 — granularity transformers @5ep (GoEmotions)      #"
echo "#   ekman6 runs double as the forward in-domain denominators#"
echo "############################################################"
run go_ekman6_bert_ep5           --dataset go_emotions --schema ekman6     --models hf_bert-base-uncased --epochs 5 --bootstrap 1000
run go_ekman6_distilbert_ep5     --dataset go_emotions --schema ekman6     --models hf_distilbert        --epochs 5 --bootstrap 1000
run go_sentiment3_bert_ep5       --dataset go_emotions --schema sentiment3 --models hf_bert-base-uncased --epochs 5 --bootstrap 1000
run go_sentiment3_distilbert_ep5 --dataset go_emotions --schema sentiment3 --models hf_distilbert        --epochs 5 --bootstrap 1000

echo
echo "############################################################"
echo "# Item 1 — forward cross-dataset transformers @5ep         #"
echo "#   GoEmotions -> SemEval-2018 (shared Ekman-6)            #"
echo "#   classical/BiLSTM cross rows reuse the old xdata run    #"
echo "############################################################"
run xdata_go2semeval_ep5 --dataset go_emotions --test-dataset sem_eval_2018_task_1 --schema ekman6 \
    --models hf_distilbert hf_bert-base-uncased --epochs 5 --bootstrap 1000

echo
echo "############################################################"
echo "# Item 2 — reverse-direction transfer @5ep                 #"
echo "#   SemEval-2018 -> GoEmotions (shared Ekman-6)            #"
echo "############################################################"
if [ "$SEMEVAL_OK" = "yes" ]; then
  # in-domain SemEval-ekman6 baseline (the reverse retention denominators)
  run sem_ekman6_ep5 --dataset sem_eval_2018_task_1 --schema ekman6 \
      --models naive_bayes logreg linsvm random_forest bilstm hf_distilbert hf_bert-base-uncased \
      --epochs 5 --bootstrap 1000
  # reverse transfer: train SemEval, test GoEmotions
  run xdata_semeval2go_ep5 --dataset sem_eval_2018_task_1 --test-dataset go_emotions --schema ekman6 \
      --models naive_bayes logreg linsvm random_forest bilstm hf_distilbert hf_bert-base-uncased \
      --epochs 5 --bootstrap 1000
else
  echo "  SKIPPED — SemEval-2018 lacks a usable train/test split (see preflight)."
  echo "  Items 1 & 3 (GoEmotions) above are unaffected."
fi

echo
echo "======================================"
echo "ALL DONE  ($(date +%H:%M:%S))"
echo "=== SUMMARY: macro_f1 per run ==="
grep -H "macro_f1=" logs/*_ep5.log 2>/dev/null
echo
echo "Next: commit results/ and push, then we recompute the 6-emotion (neutral-"
echo "excluded) retention for BOTH directions from the new *_ep5 CSVs."
