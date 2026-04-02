# EarlyExitLLM

ECE1513 Final Project - Recreating SpecEE's MLP Predictor

## Training Data Approaches

We organize training data by approach so we can compare different strategies:

```
training_data/
└── naive_all_layers/       # Approach 1: evaluate every layer, consecutive prob gaps
    ├── alpaca/
    │   ├── layer_0.csv
    │   ├── layer_1.csv
    │   └── ...layer_31.csv
    ├── gsm8k/
    ├── commonsenseqa/
    └── sst2/
```

### Approach 1: `naive_all_layers`
- Evaluates **all 32 layers** at every token step (no scheduling)
- `prob_gap` is computed between consecutive layers (layer N vs layer N-1)
- For the first layer, `prob_gap = draft_prob` (same as SpecEE source)
- Label = 1 if layer's predicted token matches the final layer's token AND is in draft top-k
- Label = 0 otherwise

**Differences from real SpecEE deployment:**
- Real SpecEE only evaluates a scheduled subset of layers (base: `[15,16,17,19,21,24,26,28]` + dynamic neighbors)
- Real SpecEE computes `prob_gap` between scheduled layers (possibly non-consecutive)
- Future approaches will address this

### Datasets

All training data is loaded from **HuggingFace train splits** to avoid data leakage
with the SpecEE benchmark eval sets (which are used for evaluation).

| Dataset | HF Source | Split | Samples | Tokens/Q | Type |
|---------|-----------|-------|--------:|---------:|------|
| alpaca | `tatsu-lab/alpaca` | train | 500 | 128 | speed |
| gsm8k | `gsm8k` (main) | train | 500 | 128 | speed |
| commonsenseqa | `tau/commonsense_qa` | train | 2,000 | 3 | accuracy |
| sst2 | `glue` (sst2) | train | 2,000 | 3 | accuracy |

- Speed datasets use the llama-2-chat prompt template
- Accuracy datasets use SpecEE's few-shot prompt formatting
- Samples are shuffled with seed=42 for reproducibility

## Scripts

### `generate_training_data.py`
Generates training data for 32 MLP predictors (one per decoder layer).

**How it works:**
1. Loads LLaMA-7B base model + EAGLE draft model
2. Loads dataset from HuggingFace train split
3. Runs inference, generating tokens per question
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
    --base-model-path meta-llama/Llama-2-7b-chat-hf \
    --draft-model-path yuhuili/EAGLE-llama2-chat-7B \
    --dataset alpaca \
    --approach naive_all_layers \
    --num-samples 500 \
    --max-new-tokens 128
```

Output goes to `training_data/<approach>/<dataset>/layer_*.csv`.

### `run_generate_data.sh`
Runs `generate_training_data.py` for all 4 datasets sequentially.

```bash
conda activate specee
bash run_generate_data.sh
```

**Output format:** Each CSV has columns:
`logit_0, logit_1, logit_2, logit_3, prob_0, prob_1, prob_2, prob_3, gap_0, gap_1, gap_2, gap_3, label`
