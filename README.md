# EarlyExitLLM

ECE1513 Final Project - Recreating SpecEE's MLP Predictor

## Repository Structure

```
EarlyExitLLM/
├── SpecEE_cloud/              # Modified SpecEE inference engine (Apache 2.0, see NOTICE)
│   ├── EE_model.py            # Main early-exit model wrapper
│   ├── model_llama_ee.py      # LLaMA model with per-layer MLP predictor + early exit logic
│   ├── EEInference.py         # Standalone inference script
│   ├── cnets.py               # EAGLE draft model
│   ├── configs.py             # EAGLE config
│   ├── accuracy_prompt.py     # Few-shot prompts for accuracy benchmarks
│   ├── benchmark/             # Evaluation datasets (mt_bench, alpaca, gsm8k, etc.)
│   ├── llama-7b/              # Original SpecEE authors' trained MLP predictors
│   ├── LICENSE.txt            # Apache 2.0 License
│   └── NOTICE                 # Attribution to original SpecEE authors
├── mlp_wahab/                 # Wahab's MLP training pipeline and trained models
├── generate_training_data.py  # Training data generation script
├── run_generate_data.sh       # Batch runner for training data generation
├── run_comparison.py          # Predictor comparison tool (speed, accuracy, all predictors)
└── README.md
```

## SpecEE_cloud (Modified)

This is a derivative of [SpecEE](https://github.com/infinigence/SpecEE) (Apache 2.0 licensed).

**Our modifications:**
- Added Apple MPS (Metal) device support for macOS inference (CUDA/MPS/CPU auto-detection)
- Integrated custom MLP predictor loading (supports both full model and state_dict `.pth` files)

The inference engine supports **both CUDA and MPS** — it auto-selects the best available device.

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
- Future approaches will address this (See ./mlp_wahab/README.md for details)

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

### `run_comparison.py`
Wrapper around SpecEE's `EEInference.py` for comparing different MLP predictors.
Includes named presets so you don't have to remember full paths.

**Available presets:**
| Name | Description | Path |
|------|-------------|------|
| `specee` | Original SpecEE authors' predictor | `SpecEE_cloud/llama-7b` |
| `jay-ap1` | Jay's naive_all_layers predictor | `trained_models/01_naive_all_layers/all` |
| `jay-ap2` | Jay's scheduled_layers predictor | `trained_models/02_scheduled_layers/all` |
| `wahab-ap1` | Wahab's approach 1 predictor | `mlp_wahab/trained_models/approach1` |
| `wahab-ap3` | Wahab's approach 3 predictor | `mlp_wahab/trained_models/approach3` |

**Single predictor usage:**
```bash
conda activate specee

# Accuracy comparison (skip HF baseline to save time)
python run_comparison.py --predictor-name wahab-ap1 --task accuracy --dataset commonsenseqa --skip-hf

# Speed benchmark
python run_comparison.py --predictor-name specee --task speed --dataset mt_bench
```

**Compare-all mode** -- runs HF baseline + all predictors across all datasets, then prints
markdown tables (Speed, Speedup, Accuracy, Avg Exit Layer):
```bash
# Full comparison
python run_comparison.py --compare-all

# Quick test with fewer samples
python run_comparison.py --compare-all --num-samples 10

# Results are also saved to comparison_results.md
```

## Findings

### KV Cache Asymmetry During Early Exit

When SpecEE triggers early exit at a given layer, the KV cache becomes **asymmetric** across layers. Layers up to and including the exit layer have KV entries for that token, but layers beyond the exit point do not.

**Example (observed via pdb):** Early exit at layer 17 (idx=17) out of 32 layers:
```
Layers 0-17:  seq_len = 167  (includes the early-exited token)
Layers 18-31: seq_len = 166  (missing the early-exited token)
```

This means:
- For all **future tokens**, attention in layers 18-31 will **not attend** to the early-exited token's representation at those layers (no K/V stored).
- This mismatch **accumulates** over time -- if multiple tokens exit at different layers, the KV cache becomes ragged with varying sequence lengths across layers.
- The paper accepts this tradeoff because the MLP predictor only exits when it's confident the token is already "correct", and the verification algorithm (checking global logits vs speculative tokens) acts as a safety net.

## Attribution

SpecEE_cloud is derived from [SpecEE](https://github.com/infinigence/SpecEE) by Jiaming Xu et al. (Copyright 2024 OpenBMB), licensed under Apache 2.0. See `SpecEE_cloud/LICENSE.txt` and `SpecEE_cloud/NOTICE` for details.
