#!/usr/bin/env bash
# M6 GPU batch: subsample-robustness for the matched-size control.
#
# The M5 matched-size result (GoEmotions cut to 6,838) used ONE random draw
# (seed 42). This re-draws the subsample at two more seeds (43, 44) so the
# data-regime reframe doesn't rest on a single lucky/unlucky sample. Combined
# with M5's seed-42 run, this gives 3 independent draws -> report mean +/- std.
#
# Changing --seed re-draws the subsample AND re-seeds model init, so each run is
# a fully independent replicate. Everything else matches M5 (6,838 rows, 5-epoch
# transformers, all 7 models, bootstrap 1000).
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
N=6838

for S in 43 44; do
  echo "== Matched-size draw: seed $S =="
  run go_ekman6_reduced_s${S}_ep5 \
      --dataset go_emotions --schema ekman6 \
      --subsample-train $N --seed $S --models $MODELS --epochs 5 --bootstrap 1000
  run xdata_go2semeval_reduced_s${S}_ep5 \
      --dataset go_emotions --test-dataset sem_eval_2018_task_1 --schema ekman6 \
      --subsample-train $N --seed $S --models $MODELS --epochs 5 --bootstrap 1000
done

echo
echo "======================================"
echo "ALL DONE  ($(date +%H:%M:%S))"
echo "=== SUMMARY: macro_f1 per run ==="
grep -H "macro_f1=" logs/*_reduced_s4*_ep5.log 2>/dev/null
echo
echo "Next: 3 draws total (M5 seed 42 + these 43/44) -> mean +/- std on the"
echo "matched-size retention and absolute cross-domain F1."
