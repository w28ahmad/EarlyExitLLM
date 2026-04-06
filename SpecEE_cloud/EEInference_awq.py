
import argparse
import os
import re
import time
from typing import Optional
from EE_model_awq import EEModel
import torch
from fastchat.model import get_conversation_template
import time
from model_llama_ee import MLP

from tqdm import trange
import json
from accuracy_prompt import get_commonsenseqa_prompt,get_mmlu_prompt,get_sst2_prompt
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def main(args):
    model = EEModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager",
        predictor_path=args.predictor_path,
        pred_thresholds = args.pred_thresholds,
    )
    if args.task == 'accuracy':
        model.eval()
        if args.dataset not in ['gsm8k','mmlu','commonsenseqa','sst2']:
            print("Dataset "+args.dataset +" is not yet supported in "+args.task+" task!")
            exit(0)
        if args.dataset == 'commonsenseqa':
            file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
            dataset = pq.read_table(file_path).to_pandas()
            correct = 0
            total = 0
            exit_layer_id_list=[]
            for _, row in dataset.iterrows():
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=model.tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).cuda()
                output_ids=model(input_ids,max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
                generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")     
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                total += 1
            accuracy = correct / total
            print(f"AWQ+SpecEE Model's accuracy on comonsenseqa is: {accuracy:.2%}")
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).cuda()
                output_ids=model.generate(input_ids,max_new_tokens=3,temperature=1e-6)
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")     
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                if predicted_answer in ['A','B','C','D','E']:
                    total += 1
            accuracy = correct / total
            print(f"HF Model's accuracy on comonsenseqa is: {accuracy:.2%}")
        elif args.dataset == 'sst2':
            file_path = "./benchmark/sst2/data/validation-00000-of-00001.parquet"  # 替换为您的数据集文件路径
            dataset = pq.read_table(file_path).to_pandas()
            exit_layer_id_list=[]
            total_time = 0
            output_ids_tot = 0
            total = 0
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = model.tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).cuda()
                seqlen = len(inputs[0])
                outputs = model(input_ids, max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
                output_ids_tot += len(outputs[0]) - seqlen
                generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
                ed = time.time()
                total_time += ed-st
                answer_start_index = len(prompt+":")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0]
                except:
                    predicted_answer = "N/A"
                if predicted_answer == label:   
                    correct += 1 
                total += 1 
            print("AWQ+SpecEE Model's accuracy on sst2 is: ",correct/total)
            torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=torch.float16,device_map="auto",attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            for _, row in dataset.iterrows():
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).cuda()
                seqlen = len(inputs[0])
                outputs = model.generate(input_ids, max_new_tokens=3,temperature=1e-6)
                output_ids_tot += len(outputs[0]) - seqlen
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                ed = time.time()
                total_time += ed-st
                answer_start_index = len(prompt+":")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0]
                except:
                    predicted_answer = "N/A"
                if predicted_answer == label:   
                    correct += 1 
                total += 1 
            print("HF Model's accuracy on sst2 is: ",correct/total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--draft-model-path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="sst2")
    parser.add_argument("--task", type=str,choices=['accuracy'], default="accuracy")
    parser.add_argument("--predictor-path", type=str, default="")
    parser.add_argument("--model-size", type=str,choices=['7B'],default="7B")
    parser.add_argument("--pred-thresholds", type=float,default=0.5)

    args = parser.parse_args()
    main(args)