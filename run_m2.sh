#!/usr/bin/env bash
# M2 GPU batch: cross-tier runs (BiLSTM + DistilBERT + BERT) on the H200 server.
# Classical tier re-measured on the server CPU (go_native_lw_server) so all
# efficiency numbers come from one machine.
cd "$(dirname "$0")"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1          # idle GPU
mkdir -p logs

run () {   # run <run-name> <args...>
  echo ">>> $1  ($(date +%H:%M:%S))"
  python scripts/run_experiment.py --dataset go_emotions "${@:2}" --run-name "$1" > "logs/$1.log" 2>&1
  grep -E "macro_f1=" "logs/$1.log" | sed "s/^/    /"
}

# Phase 1 — lightweight timing re-measured on THIS machine's CPU (native tfidf)
run go_native_lw_server --models naive_bayes logreg linsvm random_forest --schema native --bootstrap 1000

# Phase 2 — core native cross-tier (DistilBERT native already done)
run go_native_bert   --models hf_bert-base-uncased --schema native --bootstrap 1000
run go_native_bilstm --models bilstm               --schema native --bootstrap 1000

# Phase 3 — granularity sweep for the new tiers
for sch in ekman6 sentiment3; do
  run go_${sch}_bilstm     --models bilstm               --schema "$sch"
  run go_${sch}_distilbert --models hf_distilbert        --schema "$sch"
  run go_${sch}_bert       --models hf_bert-base-uncased --schema "$sch"
done

# Phase 4 — multi-seed native (3 seeds) for the stochastic tiers
run go_native_bilstm_seeds     --models bilstm               --schema native --seeds 42 43 44
run go_native_distilbert_seeds --models hf_distilbert        --schema native --seeds 42 43 44
run go_native_bert_seeds       --models hf_bert-base-uncased --schema native --seeds 42 43 44

echo "======================================"
echo "ALL DONE  ($(date +%H:%M:%S))"
echo "=== SUMMARY: macro_f1 per run ==="
grep -H "macro_f1=" logs/*.log
