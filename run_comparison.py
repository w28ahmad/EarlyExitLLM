import sys
import os

# Add SpecEE_cloud to path so imports (EE_model, model_llama_ee, accuracy_prompt, etc.) resolve
SPECEE_CLOUD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SpecEE_cloud")
sys.path.insert(0, SPECEE_CLOUD_DIR)
# Also set working directory so relative paths (./benchmark/*) work
os.chdir(SPECEE_CLOUD_DIR)

import time
import gc
import argparse
from typing import Optional
from tqdm import tqdm
import json
import torch
from accuracy_prompt import get_commonsenseqa_prompt, get_mmlu_prompt, get_sst2_prompt
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from EE_model import EEModel
from model_llama_ee import MLP
from fastchat.model import get_conversation_template

# ---------------------------------------------------------------------------
# Named predictor presets -- add new entries here as you train more models
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PREDICTOR_PRESETS = {
    "specee":    os.path.join(SPECEE_CLOUD_DIR, "llama-7b"),
    "jay-ap1":       os.path.join(PROJECT_ROOT, "trained_models", "01_naive_all_layers", "all"),
    "jay-ap2":       os.path.join(PROJECT_ROOT, "trained_models", "02_scheduled_layers", "all"),
    "wahab-ap1": os.path.join(PROJECT_ROOT, "mlp_wahab", "trained_models", "approach1"),
    "wahab-ap3": os.path.join(PROJECT_ROOT, "mlp_wahab", "trained_models", "approach3"),
}

# SPEED_DATASETS = ["mt_bench", "alpaca", "gsm8k", "sum", "qa", "humaneval"]
SPEED_DATASETS = ["mt_bench", "alpaca", "gsm8k", "qa", "humaneval"]
ACCURACY_DATASETS = ["commonsenseqa", "sst2"]
ALL_DATASETS = SPEED_DATASETS + ACCURACY_DATASETS

SYS_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, "
    "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, "
    "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased "
    "and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, "
    "explain why instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_hf_model(base_model_path, dtype, device):
    """Load HF model with proper device handling.
    On CUDA, device_map works well. On MPS, we load to CPU first then move,
    because device_map with MPS can silently load in fp32."""
    if device.type == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=dtype, device_map=device,
            attn_implementation="eager", low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=dtype,
            attn_implementation="eager", low_cpu_mem_usage=True,
        )
        model = model.to(device)
    model.eval()
    return model

def get_dtype():
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    else:
        return torch.float32

def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line:
                questions.append(json.loads(line))
    return questions[begin:end]

def make_chat_prompt(message):
    conv = get_conversation_template("llama-2-chat")
    conv.system_message = SYS_PROMPT
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt() + " "


# ---------------------------------------------------------------------------
# Evaluation functions — each returns a dict of metrics
# ---------------------------------------------------------------------------
def eval_specee_speed(model, dataset_name, num_samples, max_new_tokens, device):
    question_list = load_questions(
        f"./benchmark/{dataset_name}/question.jsonl",
        begin=0, end=num_samples if num_samples else 80,
    )
    exit_layer_id_list = []
    output_ids_tot = 0
    st = time.time()
    empty_cache()
    pbar = tqdm(range(len(question_list)), desc=f"SpecEE {dataset_name}")
    for i in pbar:
        prompt = make_chat_prompt(question_list[i]["turns"][0])
        input_ids = model.tokenizer([prompt]).input_ids
        seqlen = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).to(device)
        output_ids = model(input_ids, max_new_tokens=max_new_tokens, exit_layer_id_list=exit_layer_id_list)
        output_ids_tot += len(output_ids[0]) - seqlen
        elapsed = time.time() - st
        pbar.set_postfix(
            tok_s=f"{output_ids_tot/elapsed:.1f}",
            avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A",
        )
    ed = time.time()
    tok_s = output_ids_tot / (ed - st)
    avg_layer = sum(exit_layer_id_list) / len(exit_layer_id_list) if exit_layer_id_list else None
    return {"tok_s": tok_s, "avg_layer": avg_layer}


def eval_hf_speed(tokenizer, model, dataset_name, num_samples, max_new_tokens, device):
    question_list = load_questions(
        f"./benchmark/{dataset_name}/question.jsonl",
        begin=0, end=num_samples if num_samples else 80,
    )
    output_ids_tot = 0
    empty_cache()
    st = time.time()
    pbar = tqdm(range(len(question_list)), desc=f"HF {dataset_name}")
    for i in pbar:
        empty_cache()
        prompt = make_chat_prompt(question_list[i]["turns"][0])
        input_ids = tokenizer([prompt]).input_ids
        seqlen = len(input_ids[0])
        input_ids = torch.as_tensor(input_ids).to(device)
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        output_ids_tot += len(output_ids[0]) - seqlen
        elapsed = time.time() - st
        pbar.set_postfix(tok_s=f"{output_ids_tot/elapsed:.1f}")
    ed = time.time()
    return {"tok_s": output_ids_tot / (ed - st)}


def eval_specee_commonsenseqa(model, num_samples, device):
    file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"
    dataset = pq.read_table(file_path).to_pandas()
    if num_samples:
        dataset = dataset.head(num_samples)
    correct = 0
    total = 0
    exit_layer_id_list = []
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="SpecEE commonsenseqa")
    for _, row in pbar:
        question = row["question"]
        choices = row["choices"]
        prompt = get_commonsenseqa_prompt(question, choices["label"], choices["text"])
        input_ids = torch.as_tensor(model.tokenizer([prompt]).input_ids).to(device)
        output_ids = model(input_ids, max_new_tokens=3, exit_layer_id_list=exit_layer_id_list)
        generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            predicted = generated_text[len(prompt + "Answer:"):].strip()[0].upper()
        except:
            predicted = "N/A"
        correct_answer = row["answerKey"].strip()
        if predicted == correct_answer:
            correct += 1
        if predicted in ["A", "B", "C", "D", "E"]:
            total += 1
        pbar.set_postfix(
            acc=f"{correct/total:.2%}" if total else "N/A",
            avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A",
        )
    acc = correct / total if total else 0
    avg_layer = sum(exit_layer_id_list) / len(exit_layer_id_list) if exit_layer_id_list else None
    return {"accuracy": acc, "avg_layer": avg_layer}


def eval_hf_commonsenseqa(tokenizer, model, num_samples, device):
    file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"
    dataset = pq.read_table(file_path).to_pandas()
    if num_samples:
        dataset = dataset.head(num_samples)
    correct = 0
    total = 0
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="HF commonsenseqa")
    for _, row in pbar:
        question = row["question"]
        choices = row["choices"]
        prompt = get_commonsenseqa_prompt(question, choices["label"], choices["text"])
        input_ids = torch.as_tensor(tokenizer([prompt]).input_ids).to(device)
        output_ids = model.generate(input_ids, max_new_tokens=3, temperature=1e-6)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            predicted = generated_text[len(prompt + "Answer:"):].strip()[0].upper()
        except:
            predicted = "N/A"
        if predicted == row["answerKey"].strip():
            correct += 1
        total += 1
        pbar.set_postfix(acc=f"{correct/total:.2%}")
    return {"accuracy": correct / total if total else 0}


def eval_specee_sst2(model, num_samples, device):
    file_path = "./benchmark/sst2/data/validation-00000-of-00001.parquet"
    dataset = pq.read_table(file_path).to_pandas()
    if num_samples:
        dataset = dataset.head(num_samples)
    correct = 0
    total = 0
    exit_layer_id_list = []
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="SpecEE sst2")
    for _, row in pbar:
        sentence = row["sentence"]
        label = str(row["label"]).strip()
        prompt = get_sst2_prompt(sentence)
        input_ids = torch.as_tensor(model.tokenizer(prompt, return_tensors="pt").input_ids).to(device)
        outputs = model(input_ids, max_new_tokens=3, exit_layer_id_list=exit_layer_id_list)
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            predicted = generated_text[len(prompt + ":"):].strip()[0]
        except:
            predicted = "N/A"
        if predicted == label:
            correct += 1
        total += 1
        pbar.set_postfix(
            acc=f"{correct/total:.2%}",
            avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A",
        )
    acc = correct / total if total else 0
    avg_layer = sum(exit_layer_id_list) / len(exit_layer_id_list) if exit_layer_id_list else None
    return {"accuracy": acc, "avg_layer": avg_layer}


def eval_hf_sst2(tokenizer, model, num_samples, device):
    file_path = "./benchmark/sst2/data/validation-00000-of-00001.parquet"
    dataset = pq.read_table(file_path).to_pandas()
    if num_samples:
        dataset = dataset.head(num_samples)
    correct = 0
    total = 0
    pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="HF sst2")
    for _, row in pbar:
        sentence = row["sentence"]
        label = str(row["label"]).strip()
        prompt = get_sst2_prompt(sentence)
        input_ids = torch.as_tensor(tokenizer(prompt, return_tensors="pt").input_ids).to(device)
        outputs = model.generate(input_ids, max_new_tokens=3, temperature=1e-6)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            predicted = generated_text[len(prompt + ":"):].strip()[0]
        except:
            predicted = "N/A"
        if predicted == label:
            correct += 1
        total += 1
        pbar.set_postfix(acc=f"{correct/total:.2%}")
    return {"accuracy": correct / total if total else 0}


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------
def run_specee_on_dataset(model, dataset_name, num_samples, max_new_tokens, device):
    """Run SpecEE predictor on a single dataset. Returns dict of metrics."""
    model.eval()
    if dataset_name in SPEED_DATASETS:
        return eval_specee_speed(model, dataset_name, num_samples, max_new_tokens, device)
    elif dataset_name == "commonsenseqa":
        return eval_specee_commonsenseqa(model, 8*num_samples, device)
    elif dataset_name == "sst2":
        return eval_specee_sst2(model, 8*num_samples, device)
    return {}


def run_hf_on_dataset(tokenizer, model, dataset_name, num_samples, max_new_tokens, device):
    """Run HF baseline on a single dataset. Returns dict of metrics."""
    if dataset_name in SPEED_DATASETS:
        return eval_hf_speed(tokenizer, model, dataset_name, num_samples, max_new_tokens, device)
    elif dataset_name == "commonsenseqa":
        return eval_hf_commonsenseqa(tokenizer, model, num_samples, device)
    elif dataset_name == "sst2":
        return eval_hf_sst2(tokenizer, model, num_samples, device)
    return {}


# ---------------------------------------------------------------------------
# Single-predictor mode (original behavior)
# ---------------------------------------------------------------------------
def main_single(args):
    device = get_device()
    dtype = get_dtype()

    print(f"\n{'='*60}")
    print(f"Predictor: {args.predictor_name or 'custom'}")
    print(f"Path:      {args.predictor_path}")
    print(f"Task:      {args.task}  |  Dataset: {args.dataset}")
    print(f"Threshold: {args.pred_thresholds}")
    print(f"Device:    {device}")
    print(f"{'='*60}\n")

    model = EEModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        attn_implementation="eager",
        predictor_path=args.predictor_path,
        pred_thresholds=args.pred_thresholds,
    )

    result = run_specee_on_dataset(model, args.dataset, args.num_samples, args.max_new_tokens, device)
    if "tok_s" in result:
        print(f"SpecEE {args.dataset} tok/s: {result['tok_s']:.2f}")
    if "accuracy" in result:
        print(f"SpecEE {args.dataset} accuracy: {result['accuracy']:.2%}")
    if result.get("avg_layer") is not None:
        print(f"average layer: {result['avg_layer']:.1f}")

    if not args.skip_hf:
        del model
        gc.collect()
        empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        hf_model = load_hf_model(args.base_model_path, dtype, device)
        hf_result = run_hf_on_dataset(tokenizer, hf_model, args.dataset, args.num_samples, args.max_new_tokens, device)
        if "tok_s" in hf_result:
            print(f"HF {args.dataset} tok/s: {hf_result['tok_s']:.2f}")
            print(f"SpecEE acceleration ratio: {result['tok_s']/hf_result['tok_s']:.2f}x")
        if "accuracy" in hf_result:
            print(f"HF {args.dataset} accuracy: {hf_result['accuracy']:.2%}")


# ---------------------------------------------------------------------------
# Compare-all mode
# ---------------------------------------------------------------------------
def main_compare_all(args):
    device = get_device()
    dtype = get_dtype()

    datasets_to_run = list(ALL_DATASETS)

    print(f"\n{'='*60}")
    print(f"COMPARE-ALL MODE")
    print(f"Predictors: HF (baseline), {', '.join(PREDICTOR_PRESETS.keys())}")
    print(f"Datasets:   {', '.join(datasets_to_run)}")
    print(f"Threshold:  {args.pred_thresholds}")
    print(f"Device:     {device}")
    if args.num_samples:
        print(f"Samples:    {args.num_samples}")
    print(f"{'='*60}\n")

    # results[predictor_name][dataset_name] = {tok_s, accuracy, avg_layer}
    results = {}

    # --- 1. HF Baseline (no predictor) ---
    print(f"\n{'─'*60}")
    print(f"Running: HF Baseline (no early exit)")
    print(f"{'─'*60}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=dtype, device_map=device,
        attn_implementation="eager", low_cpu_mem_usage=True,
    )
    hf_model.eval()
    results["HF Baseline"] = {}
    for ds in datasets_to_run:
        empty_cache()
        r = run_hf_on_dataset(tokenizer, hf_model, ds, args.num_samples, args.max_new_tokens, device)
        results["HF Baseline"][ds] = r
        print(f"  {ds}: {r}")
    del hf_model, tokenizer
    gc.collect()
    empty_cache()

    # --- 2. Each predictor ---
    for pred_name, pred_path in PREDICTOR_PRESETS.items():
        print(f"\n{'─'*60}")
        print(f"Running: {pred_name}  ({pred_path})")
        print(f"{'─'*60}")
        model = EEModel.from_pretrained(
            base_model_path=args.base_model_path,
            ea_model_path=args.draft_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device,
            attn_implementation="eager",
            predictor_path=pred_path,
            pred_thresholds=args.pred_thresholds,
        )
        results[pred_name] = {}
        for ds in datasets_to_run:
            empty_cache()
            r = run_specee_on_dataset(model, ds, args.num_samples, args.max_new_tokens, device)
            results[pred_name][ds] = r
            print(f"  {ds}: {r}")
        del model
        gc.collect()
        empty_cache()

    # --- 3. Build & print tables ---
    predictor_names = ["HF Baseline"] + list(PREDICTOR_PRESETS.keys())
    hf_speed = {ds: results["HF Baseline"][ds].get("tok_s") for ds in SPEED_DATASETS}

    # -- Speed table --
    print("\n")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    speed_header = "| Predictor | " + " | ".join(SPEED_DATASETS) + " |"
    speed_sep = "|---|" + "|".join(["---"] * len(SPEED_DATASETS)) + "|"

    print("\n### Speed (tokens/s)")
    print()
    print(speed_header)
    print(speed_sep)
    for name in predictor_names:
        cells = []
        for ds in SPEED_DATASETS:
            r = results[name].get(ds, {})
            tok = r.get("tok_s")
            cells.append(f"{tok:.2f}" if tok is not None else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()

    # -- Speedup table --
    print("### Speedup (vs HF Baseline)")
    print()
    print(speed_header)
    print(speed_sep)
    for name in predictor_names:
        cells = []
        for ds in SPEED_DATASETS:
            tok = results[name].get(ds, {}).get("tok_s")
            base = hf_speed.get(ds)
            if tok is not None and base is not None and base > 0:
                cells.append(f"{tok/base:.2f}x")
            else:
                cells.append("-")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()

    # -- Accuracy table --
    acc_header = "| Predictor | " + " | ".join(ACCURACY_DATASETS) + " |"
    acc_sep = "|---|" + "|".join(["---"] * len(ACCURACY_DATASETS)) + "|"

    print("### Accuracy")
    print()
    print(acc_header)
    print(acc_sep)
    for name in predictor_names:
        cells = []
        for ds in ACCURACY_DATASETS:
            r = results[name].get(ds, {})
            acc = r.get("accuracy")
            cells.append(f"{acc:.2%}" if acc is not None else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()

    # -- Average exit layer table (SpecEE predictors only) --
    ee_names = list(PREDICTOR_PRESETS.keys())
    layer_header = "| Predictor | " + " | ".join(ALL_DATASETS) + " |"
    layer_sep = "|---|" + "|".join(["---"] * len(ALL_DATASETS)) + "|"

    print("### Avg Exit Layer (lower = more aggressive early exit)")
    print()
    print(layer_header)
    print(layer_sep)
    for name in ee_names:
        cells = []
        for ds in ALL_DATASETS:
            r = results[name].get(ds, {})
            avg = r.get("avg_layer")
            cells.append(f"{avg:.1f}" if avg is not None else "-")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()

    # -- Save to file --
    out_path = os.path.join(PROJECT_ROOT, "comparison_results.md")
    with open(out_path, "w") as f:
        f.write(f"# SpecEE Predictor Comparison\n\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Threshold: {args.pred_thresholds}\n")
        f.write(f"- Base model: {args.base_model_path}\n")
        f.write(f"- Draft model: {args.draft_model_path}\n")
        if args.num_samples:
            f.write(f"- Samples per dataset: {args.num_samples}\n")
        f.write(f"\n## Speed (tokens/s)\n\n")
        f.write(speed_header + "\n")
        f.write(speed_sep + "\n")
        for name in predictor_names:
            cells = []
            for ds in SPEED_DATASETS:
                tok = results[name].get(ds, {}).get("tok_s")
                cells.append(f"{tok:.2f}" if tok is not None else "-")
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
        f.write(f"\n## Speedup (vs HF Baseline)\n\n")
        f.write(speed_header + "\n")
        f.write(speed_sep + "\n")
        for name in predictor_names:
            cells = []
            for ds in SPEED_DATASETS:
                tok = results[name].get(ds, {}).get("tok_s")
                base = hf_speed.get(ds)
                if tok is not None and base is not None and base > 0:
                    cells.append(f"{tok/base:.2f}x")
                else:
                    cells.append("-")
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
        f.write(f"\n## Accuracy\n\n")
        f.write(acc_header + "\n")
        f.write(acc_sep + "\n")
        for name in predictor_names:
            cells = []
            for ds in ACCURACY_DATASETS:
                acc = results[name].get(ds, {}).get("accuracy")
                cells.append(f"{acc:.2%}" if acc is not None else "-")
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
        f.write(f"\n## Avg Exit Layer\n\n")
        f.write(layer_header + "\n")
        f.write(layer_sep + "\n")
        for name in ee_names:
            cells = []
            for ds in ALL_DATASETS:
                avg = results[name].get(ds, {}).get("avg_layer")
                cells.append(f"{avg:.1f}" if avg is not None else "-")
            f.write(f"| {name} | " + " | ".join(cells) + " |\n")
        f.write("\n")
    print(f"Results also saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecEE predictor comparison tool")
    parser.add_argument("--base-model-path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--draft-model-path", type=str, default="yuhuili/EAGLE-llama2-chat-7B")
    parser.add_argument("--dataset", type=str, default="mt_bench")
    parser.add_argument("--task", type=str, choices=["speed", "accuracy"], default="speed")
    parser.add_argument("--model-size", type=str, choices=["7B"], default="7B")
    parser.add_argument("--pred-thresholds", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples per dataset (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens for speed tasks (default: 256)")
    parser.add_argument("--skip-hf", action="store_true", help="Skip HF baseline (single-predictor mode only)")

    # Predictor selection
    parser.add_argument("--predictor-name", type=str, choices=list(PREDICTOR_PRESETS.keys()),
                        default=None, help=f"Named predictor preset: {list(PREDICTOR_PRESETS.keys())}")
    parser.add_argument("--predictor-path", type=str, default="",
                        help="Custom predictor path (overrides --predictor-name)")

    # Compare-all mode
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all predictors on all datasets and print comparison tables")

    args = parser.parse_args()

    if args.compare_all:
        main_compare_all(args)
    else:
        # Resolve predictor path for single mode
        if args.predictor_path:
            pass
        elif args.predictor_name:
            args.predictor_path = PREDICTOR_PRESETS[args.predictor_name]
        else:
            parser.error("Either --predictor-name, --predictor-path, or --compare-all is required")
        main_single(args)
