#!/bin/bash
# Generate training data from HuggingFace train splits (no eval data leakage).
# Usage: bash run_generate_data.sh
#
# Speed datasets (128 tokens/question): alpaca, gsm8k
# Accuracy datasets (3 tokens/question): commonsenseqa, sst2

BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
DRAFT_MODEL="yuhuili/EAGLE-llama2-chat-7B"
APPROACH="naive_all_layers"

# Speed datasets: 500 samples x 128 tokens each
for DATASET in "alpaca" "gsm8k"; do
    echo "============================================"
    echo "Generating data for: ${DATASET}"
    echo "============================================"
    python generate_training_data.py \
        --base-model-path "${BASE_MODEL}" \
        --draft-model-path "${DRAFT_MODEL}" \
        --dataset "${DATASET}" \
        --approach "${APPROACH}" \
        --num-samples 500 \
        --max-new-tokens 128
    echo ""
done

# Accuracy datasets: 2000 samples x 3 tokens each
for DATASET in "commonsenseqa" "sst2"; do
    echo "============================================"
    echo "Generating data for: ${DATASET}"
    echo "============================================"
    python generate_training_data.py \
        --base-model-path "${BASE_MODEL}" \
        --draft-model-path "${DRAFT_MODEL}" \
        --dataset "${DATASET}" \
        --approach "${APPROACH}" \
        --num-samples 2000
    echo ""
done

echo "All datasets done!"
