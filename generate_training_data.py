"""
Generate training data for 32 MLP predictors (one per layer) for SpecEE.

For each token generation step, this script:
1. Runs through ALL 32 layers (no early exit)
2. At each layer, collects 12 features:
   - 4 draft logits (top-k logits from draft model's lm_head)
   - 4 local probabilities (softmax of draft logits)
   - 4 probability variations (current prob - previous layer prob)
3. Labels each layer: 1 if the token predicted at that layer matches
   the draft model's top-k candidates AND matches the final layer's token,
   0 otherwise.

Output: 32 CSV files (one per layer), each row = 12 features + 1 label.

Usage:
    conda activate specee
    python generate_training_data.py \
        --base-model-path <path-to-llama-7b> \
        --draft-model-path <path-to-eagle-model> \
        --dataset mt_bench \
        --max-new-tokens 128 \
        --output-dir ./training_data
"""

import argparse
import os
import sys
import json
import csv
import gc
from typing import Optional, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add SpecEE_cloud to path so we can import its modules
SPECEE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SpecEE', 'SpecEE_cloud')
sys.path.insert(0, SPECEE_DIR)

from model_llama_ee import LlamaForCausalLM as LlamaForCausalLMEE
from configs import EConfig
from cnets import Model
from transformers import AutoConfig, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.cache_utils import DynamicCache

TOP_K = 4  # Same as SpecEE's cnets.py


def load_questions(question_file: str, begin: int = 0, end: int = None):
    """Load questions from a JSONL file."""
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions[begin:end]


def format_prompt(message: str) -> str:
    """Format a message into a llama-2-chat prompt."""
    sys_p = (
        "You are a helpful, respectful and honest assistant. Always answer as "
        "helpfully as possible, while being safe. Your answers should not include "
        "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why "
        "instead of answering something not correct. If you don't know the answer to a "
        "question, please don't share false information."
    )
    prompt = f"[INST] <<SYS>>\n{sys_p}\n<</SYS>>\n\n{message} [/INST] "
    return prompt


class DataCollector:
    """Appends features and labels to per-layer CSV files incrementally."""

    HEADER = [
        'logit_0', 'logit_1', 'logit_2', 'logit_3',
        'prob_0', 'prob_1', 'prob_2', 'prob_3',
        'gap_0', 'gap_1', 'gap_2', 'gap_3',
        'label'
    ]

    def __init__(self, num_layers: int, output_dir: str):
        self.num_layers = num_layers
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sample_counts = [0] * num_layers
        self.positive_counts = [0] * num_layers

        # Write headers to all CSV files
        for layer_idx in range(num_layers):
            filepath = os.path.join(output_dir, f'layer_{layer_idx}.csv')
            with open(filepath, 'w', newline='') as f:
                csv.writer(f).writerow(self.HEADER)

        # Keep file handles open for appending
        self._files = []
        self._writers = []
        for layer_idx in range(num_layers):
            filepath = os.path.join(output_dir, f'layer_{layer_idx}.csv')
            fh = open(filepath, 'a', newline='')
            self._files.append(fh)
            self._writers.append(csv.writer(fh))

    def add_sample(self, layer_idx: int, features: list, label: int):
        self._writers[layer_idx].writerow(features + [label])
        self.sample_counts[layer_idx] += 1
        self.positive_counts[layer_idx] += label

    def flush(self):
        for fh in self._files:
            fh.flush()

    def close(self):
        for fh in self._files:
            fh.close()

    def print_summary(self):
        print("\n--- Label Distribution Summary ---")
        for layer_idx in range(self.num_layers):
            n_total = self.sample_counts[layer_idx]
            n_pos = self.positive_counts[layer_idx]
            if n_total == 0:
                continue
            ratio = n_pos / n_total
            print(f"  Layer {layer_idx:2d}: {n_total:6d} samples, {n_pos:6d} positive ({ratio:.1%})")


def collect_features_and_labels(
    layers, norm_fn, lm_head, embed_tokens,
    token, past_key_values,
    draft_lm_head_weight, draft_token_index,
    collector,
):
    """
    Run token through all layers, collecting features/labels at each.
    Memory-efficient: computes features inline, no storing all hidden states.

    Returns: (final_hidden, final_token, past_key_values)
    """
    h = embed_tokens(token)
    device = token.device
    num_layers = len(layers)

    # Prepare attention mask
    use_legacy_cache = not isinstance(past_key_values, DynamicCache)
    if use_legacy_cache:
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    past_len = past_key_values.get_seq_length()
    batch_size, seq_length = token.shape[:2]
    position_ids = torch.arange(
        past_len, seq_length + past_len, dtype=torch.long, device=device
    ).unsqueeze(0)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (batch_size, seq_length), h, past_len
    )

    # --- Pass 1: Forward through all layers, store per-layer features ---
    # We store features (small: 12 floats) and the argmax token id per layer.
    # Then after we know the final token, we assign labels.
    layer_features = []  # list of (12 floats)
    layer_tokens = []    # token predicted at each layer

    last_prob = None

    for idx, decoder_layer in enumerate(layers):
        layer_outputs = decoder_layer(
            h,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            output_attentions=False,
            use_cache=True,
        )
        h = layer_outputs[0]

        # Compute features at this layer (only last token position to save memory)
        h_normed = norm_fn(h)
        # Only use last token position: [1, 1, hidden] -> reduces lm_head memory
        h_last = h_normed[:, -1:, :]
        draft_logits = F.linear(h_last, draft_lm_head_weight)  # [1, 1, top_k]
        draft_prob = F.softmax(draft_logits, dim=-1)

        if last_prob is None:
            prob_gap = draft_prob
        else:
            prob_gap = draft_prob - last_prob
        last_prob = draft_prob

        # Extract features (12 floats)
        feat = (
            draft_logits[0, 0, :].detach().cpu().tolist() +
            draft_prob[0, 0, :].detach().cpu().tolist() +
            prob_gap[0, 0, :].detach().cpu().tolist()
        )
        layer_features.append(feat)

        # Get token this layer would predict
        layer_logits = lm_head(h_last)
        layer_tok = torch.argmax(layer_logits[0, 0]).item()
        layer_tokens.append(layer_tok)

        # Free intermediate tensors immediately
        del h_normed, h_last, draft_logits, draft_prob, prob_gap, layer_logits

    # Final token is from last layer
    final_token = layer_tokens[-1]
    final_in_draft = (final_token in draft_token_index.tolist()[0]
                      if draft_token_index.dim() == 2
                      else final_token in draft_token_index.tolist())

    # Assign labels and store
    for idx in range(num_layers):
        if final_in_draft and layer_tokens[idx] == final_token:
            label = 1
        else:
            label = 0
        collector.add_sample(idx, layer_features[idx], label)

    # Return final hidden state and token for next step
    final_hidden = norm_fn(h)
    final_token_tensor = torch.tensor([[final_token]], device=device, dtype=torch.long)

    del layer_features, layer_tokens
    return final_hidden, final_token_tensor, past_key_values


def generate_data(args):
    """Main data generation loop."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load base model
    print("Loading base model...")
    base_model = LlamaForCausalLMEE.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
    )
    base_model.eval()
    num_layers = len(base_model.model.layers)
    print(f"Model has {num_layers} layers")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    # Load EAGLE draft model on CPU to save GPU memory
    # (base model ~13.5GB + EAGLE ~1.5GB would exceed 16GB GPU)
    print("Loading EAGLE draft model (on CPU to save GPU memory)...")
    config = EConfig.from_pretrained(args.draft_model_path)
    bias = getattr(config, "bias", True)
    ea_layer = Model(config, bias=bias)

    ea_layer.diff_device = False
    ea_layer.to(base_model.dtype).to("cpu")

    # We need a CPU copy of lm_head weights for EAGLE's topK_genrate
    lm_head_cpu = torch.nn.Linear(
        base_model.lm_head.in_features, base_model.lm_head.out_features, bias=False
    ).to(base_model.dtype).to("cpu")
    lm_head_cpu.weight.data = base_model.lm_head.weight.data.cpu().clone()

    # Load draft model weights
    if os.path.isdir(args.draft_model_path):
        load_model_path = os.path.join(args.draft_model_path, "pytorch_model.bin")
    else:
        from huggingface_hub import hf_hub_download
        load_model_path = hf_hub_download(repo_id=args.draft_model_path, filename="pytorch_model.bin")

    ea_layer_state_dict = torch.load(load_model_path, map_location="cpu", weights_only=False)
    ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
    ea_layer.eval()

    gpu_device = base_model.model.layers[-1].self_attn.q_proj.weight.device

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    benchmark_dir = os.path.join(SPECEE_DIR, 'benchmark', args.dataset)
    question_file = os.path.join(benchmark_dir, 'question.jsonl')
    questions = load_questions(question_file, begin=args.begin, end=args.end)
    print(f"Loaded {len(questions)} questions")

    # Initialize data collector
    collector = DataCollector(num_layers, args.output_dir)

    # Generate data
    print("Generating training data...")
    with torch.inference_mode():
        for q_idx in tqdm(range(len(questions)), desc="Questions"):
            message = questions[q_idx]['turns'][0]
            prompt = format_prompt(message)
            input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(gpu_device)

            ea_layer.reset_kv()
            torch.cuda.empty_cache()

            # --- First token (init=True): run full model, no features collected ---
            outputs, token = base_model.model(
                input_ids=input_ids,
                use_cache=True,
                lm_head=base_model.lm_head,
                exit_layer_id_list=[],
                init=True,
            )
            hidden_states = outputs[0].clone()
            past_key_values = outputs[1]
            token = token.to(gpu_device)
            input_ids = torch.cat((input_ids, token), dim=1)

            # Get draft model's top-k (EAGLE runs on CPU)
            topk_index, topk_prob, top_head_weight = ea_layer.topK_genrate(
                hidden_states.cpu(), input_ids.cpu(), lm_head_cpu
            )
            # Move results to GPU for feature computation
            topk_index_gpu = topk_index.to(gpu_device)
            top_head_weight_gpu = top_head_weight.to(gpu_device)

            # --- Subsequent tokens: collect features at each layer ---
            for step in range(args.max_new_tokens - 1):
                hidden_states, token, past_key_values = collect_features_and_labels(
                    layers=base_model.model.layers,
                    norm_fn=base_model.model.norm,
                    lm_head=base_model.lm_head,
                    embed_tokens=base_model.model.embed_tokens,
                    token=token,
                    past_key_values=past_key_values,
                    draft_lm_head_weight=top_head_weight_gpu,
                    draft_token_index=topk_index_gpu,
                    collector=collector,
                )

                input_ids = torch.cat((input_ids, token.to(gpu_device)), dim=1)

                # Get draft model's top-k (EAGLE on CPU)
                topk_index, topk_prob, top_head_weight = ea_layer.topK_genrate(
                    hidden_states.cpu(), input_ids.cpu(), lm_head_cpu
                )
                topk_index_gpu = topk_index.to(gpu_device)
                top_head_weight_gpu = top_head_weight.to(gpu_device)

                # Check for EOS
                if tokenizer.eos_token_id in input_ids[0, -args.max_new_tokens:].tolist():
                    break

            # Flush to disk after each question
            collector.flush()

            # Free KV cache between questions
            del past_key_values, hidden_states, outputs, token, input_ids
            del topk_index, topk_prob, top_head_weight
            del topk_index_gpu, top_head_weight_gpu
            gc.collect()
            torch.cuda.empty_cache()

    collector.close()
    collector.print_summary()
    print(f"\nDone! Training data saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for SpecEE MLP predictors")
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path to LLaMA-7B base model")
    parser.add_argument("--draft-model-path", type=str, required=True,
                        help="Path to EAGLE draft model")
    parser.add_argument("--dataset", type=str, default="mt_bench",
                        choices=["mt_bench", "alpaca", "gsm8k", "sum", "qa", "humaneval"],
                        help="Dataset to use for data generation")
    parser.add_argument("--output-dir", type=str, default="./training_data",
                        help="Directory to save per-layer CSV files")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per prompt")
    parser.add_argument("--begin", type=int, default=0,
                        help="Start index in dataset")
    parser.add_argument("--end", type=int, default=None,
                        help="End index in dataset (None = all)")
    args = parser.parse_args()
    generate_data(args)
