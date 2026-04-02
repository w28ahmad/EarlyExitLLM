# EarlyExitLLM

ECE1513 Final Project - Recreating SpecEE's MLP Predictor

## Scripts

### `generate_training_data.py`
Generates training data for 32 MLP predictors (one per decoder layer).

**How it works:**
1. Loads LLaMA-7B base model + EAGLE draft model
2. Runs inference on a dataset (e.g., mt_bench), generating tokens
3. At each token generation step, runs through ALL 32 layers without early exit
4. At each layer, extracts 12 features:
   - 4 draft logits (projection of hidden states onto draft model's top-k head weights)
   - 4 local probabilities (softmax of draft logits)
   - 4 probability variations (current prob - previous layer's prob)
5. Labels each layer: `1` if exiting here produces the correct token, `0` otherwise
6. Saves 32 CSV files (`layer_0.csv` ... `layer_31.csv`)

**Usage:**
```bash
conda activate specee
python generate_training_data.py \
    --base-model-path <path-to-llama-7b> \
    --draft-model-path <path-to-eagle-model> \
    --dataset mt_bench \
    --max-new-tokens 128 \
    --output-dir ./training_data
```

**Output format:** Each CSV has columns:
`logit_0, logit_1, logit_2, logit_3, prob_0, prob_1, prob_2, prob_3, gap_0, gap_1, gap_2, gap_3, label`
