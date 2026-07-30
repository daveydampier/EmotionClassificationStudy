#!/usr/bin/env bash
# M5 GPU batch: matched-size CONTROL for the cross-dataset robustness finding.
#
# Question it answers: is the lightweight forward-transfer advantage driven by
# GoEmotions being a LARGER source (43k vs SemEval's 6.8k), or by it being a
# richer/different DOMAIN? We subsample GoEmotions train down to SemEval's size
# (6,838, random) and re-run the forward transfer. Then:
#
#   retention_reduced (this batch) vs retention_full (the M4 xdata_go2semeval_ep5)
#     - if lightweight STILL retains better at matched size -> it's DOMAIN/richness
#     - if the advantage shrinks/vanishes            -> it's SOURCE SIZE
#
# Two runs, both at matched size (6,838), 5-epoch transformers, seed 42 to match
# the single-seed full-size forward run:
#   go_ekman6_reduced_ep5        = in-domain denominator (train 6.8k GoE, test GoE)
#   xdata_go2semeval_reduced_ep5 = forward cross (train 6.8k GoE, test SemEval)
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

MODELS="naive_bayes logreg linsvm random_forest bilstm hf_distilbert hf_bert-base-uncased"
N=6838   # SemEval-2018 train size — the size we match to

echo "== Matched-size control: GoEmotions train randomly subsampled to $N =="
run go_ekman6_reduced_ep5 \
    --dataset go_emotions --schema ekman6 \
    --subsample-train $N --models $MODELS --epochs 5 --bootstrap 1000
run xdata_go2semeval_reduced_ep5 \
    --dataset go_emotions --test-dataset sem_eval_2018_task_1 --schema ekman6 \
    --subsample-train $N --models $MODELS --epochs 5 --bootstrap 1000

echo
echo "======================================"
echo "ALL DONE  ($(date +%H:%M:%S))"
echo "=== SUMMARY: macro_f1 per run ==="
grep -H "macro_f1=" logs/*_reduced_ep5.log 2>/dev/null
echo
echo "Next: push results, then compare retention_reduced vs retention_full to"
echo "settle SOURCE SIZE vs SOURCE DOMAIN."
