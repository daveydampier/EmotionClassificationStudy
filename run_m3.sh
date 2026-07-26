#!/usr/bin/env bash
# M3 GPU batch: transformer training-budget check (5/8 epochs), cross-dataset
# generalization (GoEmotions -> SemEval-2018), and multi-seed granularity.
cd "$(dirname "$0")"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
mkdir -p logs
run () { echo ">>> $1 ($(date +%H:%M:%S))"; python scripts/run_experiment.py --dataset go_emotions "${@:2}" --run-name "$1" > "logs/$1.log" 2>&1; grep -E "macro_f1=" "logs/$1.log" | sed "s/^/    /"; }

# P1 — transformer training-budget check (native, 5 & 8 epochs)
run go_native_bert_ep5       --models hf_bert-base-uncased --schema native --epochs 5 --bootstrap 1000
run go_native_bert_ep8       --models hf_bert-base-uncased --schema native --epochs 8 --bootstrap 1000
run go_native_distilbert_ep5 --models hf_distilbert        --schema native --epochs 5 --bootstrap 1000
run go_native_distilbert_ep8 --models hf_distilbert        --schema native --epochs 8 --bootstrap 1000

# P2 — cross-dataset generalization (train GoEmotions, test SemEval-2018, shared Ekman-6)
run xdata_go2semeval --models naive_bayes logreg linsvm random_forest bilstm hf_distilbert hf_bert-base-uncased \
    --schema ekman6 --test-dataset sem_eval_2018_task_1 --bootstrap 1000

# P3 — multi-seed granularity for the stochastic tiers (ekman6 + sentiment3)
for sch in ekman6 sentiment3; do
  run go_${sch}_bilstm_seeds     --models bilstm               --schema $sch --seeds 42 43 44
  run go_${sch}_distilbert_seeds --models hf_distilbert        --schema $sch --seeds 42 43 44
  run go_${sch}_bert_seeds       --models hf_bert-base-uncased --schema $sch --seeds 42 43 44
done

# Canonical 5-epoch transformers with seed error bars (run separately after the batch)
run go_native_transformers_ep5_seeds --models hf_bert-base-uncased hf_distilbert \
    --schema native --epochs 5 --seeds 42 43 44 --bootstrap 1000

echo "==== DONE ($(date +%H:%M:%S)) ===="
echo "--- epochs check ---"; grep -H "macro_f1=" logs/go_native_*_ep*.log
