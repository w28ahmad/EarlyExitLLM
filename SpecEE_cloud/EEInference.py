import time
import gc
import argparse
from typing import Optional
from tqdm import trange, tqdm
import json
import torch
from accuracy_prompt import get_commonsenseqa_prompt,get_mmlu_prompt,get_sst2_prompt
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer,AutoModelForCausalLM
from EE_model import EEModel
from model_llama_ee import MLP
from fastchat.model import get_conversation_template

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def main(args):
    device = get_device()
    dtype = get_dtype()
    model = EEModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device,
        attn_implementation="eager",
        predictor_path=args.predictor_path,
        pred_thresholds = args.pred_thresholds,
    )
    if args.task == 'speed':
        if args.dataset not in ['alpaca','gsm8k','mt_bench','sum','qa','humaneval']:
            print("Dataset "+args.dataset +" is not yet supported in "+args.task+" task!")
            exit(0)
        question_list = load_questions('./benchmark/'+args.dataset+'/question.jsonl',begin=0,end=args.num_samples if args.num_samples else 80)
        exit_layer_id_list=[]
        output_ids_tot = 0
        st = time.time()
        empty_cache()
        pbar = tqdm(range(len(question_list)), desc="SpecEE speed")
        for i in pbar:
            message = question_list[i]['turns'][0]
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids=model.tokenizer([prompt]).input_ids
            seqlen = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).to(device)
            output_ids=model(input_ids,max_new_tokens=args.max_new_tokens,exit_layer_id_list=exit_layer_id_list)
            output_ids_tot += len(output_ids[0]) - seqlen
            output=model.tokenizer.decode(output_ids[0])
            elapsed = time.time() - st
            pbar.set_postfix(tok_s=f"{output_ids_tot/elapsed:.1f}",
                             avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A")
        ed = time.time()
        spec = output_ids_tot/(ed-st)
        print('SpecEE '+ args.dataset + ' tokens per second :  ',spec)
        print('average layer :  ',sum(exit_layer_id_list)/len(exit_layer_id_list))
        del model
        gc.collect()
        empty_cache()
        # tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        # model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=dtype,device_map=device,attn_implementation="eager",low_cpu_mem_usage=True)
        # model.eval()
        # output_ids_tot = 0
        # empty_cache()
        # st = time.time()
        # pbar = tqdm(range(len(question_list)), desc="HF speed")
        # for i in pbar:
        #     empty_cache()
        #     message = question_list[i]['turns'][0]
        #     conv = get_conversation_template("llama-2-chat")
        #     sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        #     conv.system_message = sys_p
        #     conv.append_message(conv.roles[0], message)
        #     conv.append_message(conv.roles[1], None)
        #     prompt = conv.get_prompt() + " "
        #     input_ids=tokenizer([prompt]).input_ids
        #     seqlen = len(input_ids[0])
        #     input_ids = torch.as_tensor(input_ids).to(device)
        #     output_ids=model.generate(input_ids,max_new_tokens=args.max_new_tokens,do_sample=False)
        #     output_ids_tot += len(output_ids[0]) - seqlen
        #     output=tokenizer.decode(output_ids[0])
        #     elapsed = time.time() - st
        #     pbar.set_postfix(tok_s=f"{output_ids_tot/elapsed:.1f}")
        # ed = time.time()
        # hf = output_ids_tot/(ed-st)
        # print('HF '+args.dataset + ' tokens per second :  ',hf)
        # print('SpecEE acceleration ratio is: ',spec/hf)
    elif args.task == 'accuracy':
        model.eval()
        if args.dataset not in ['commonsenseqa','sst2']:
            print("Dataset "+args.dataset +" is not yet supported in "+args.task+" task!")
            exit(0)
        if args.dataset == 'commonsenseqa':
            file_path = "./benchmark/commonsense_qa/data/validation-00000-of-00001.parquet"
            dataset = pq.read_table(file_path).to_pandas()
            if args.num_samples:
                dataset = dataset.head(args.num_samples)
            correct = 0
            total = 0
            exit_layer_id_list=[]
            pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="SpecEE commonsenseqa")
            for _, row in pbar:
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=model.tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).to(device)
                output_ids=model(input_ids,max_new_tokens=3,exit_layer_id_list=exit_layer_id_list)
                generated_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                if predicted_answer in ['A','B','C','D','E']:
                    total += 1
                pbar.set_postfix(acc=f"{correct/total:.2%}" if total else "N/A",
                                 avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A")
            accuracy = correct / total
            print(f"SpecEE Model's accuracy on comonsenseqa is: {accuracy:.2%}")
            print('average layer :  ',sum(exit_layer_id_list)/len(exit_layer_id_list)) # added by Jay
            del model
            gc.collect()
            empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=dtype,device_map=device,attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="HF commonsenseqa")
            for _, row in pbar:
                question = row['question']
                choices = row['choices']
                options = choices['label']
                answers = choices['text']
                correct_answer = row['answerKey'].strip()
                prompt = get_commonsenseqa_prompt(question,options,answers)
                input_ids=tokenizer([prompt]).input_ids
                input_ids = torch.as_tensor(input_ids).to(device)
                output_ids=model.generate(input_ids,max_new_tokens=3,temperature=1e-6)
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                answer_start_index = len(prompt+"Answer:")
                try:
                    predicted_answer = generated_text[answer_start_index:].strip()[0].upper()
                except:
                    predicted_answer = "N/A"
                if predicted_answer == correct_answer:
                    correct += 1
                # if predicted_answer in ['A','B','C','D','E']:
                total += 1
                pbar.set_postfix(acc=f"{correct/total:.2%}")
            accuracy = correct / total
            print(f"HF Model's accuracy on comonsenseqa is: {accuracy:.2%}")
        elif args.dataset == 'sst2':
            file_path = "./benchmark/sst2/data/validation-00000-of-00001.parquet"
            dataset = pq.read_table(file_path).to_pandas()
            if args.num_samples:
                dataset = dataset.head(args.num_samples)
            exit_layer_id_list=[]
            total_time = 0
            output_ids_tot = 0
            total = 0
            correct = 0
            total = 0
            pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="SpecEE sst2")
            for _, row in pbar:
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = model.tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).to(device)
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
                total +=1
                pbar.set_postfix(acc=f"{correct/total:.2%}",
                                 avg_layer=f"{sum(exit_layer_id_list)/len(exit_layer_id_list):.1f}" if exit_layer_id_list else "N/A")
            print("SpecEE Model's accuracy on sst2 is: ",correct/total)
            print('average layer :  ',sum(exit_layer_id_list)/len(exit_layer_id_list)) # added by Jay
            del model
            gc.collect()
            empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
            model = AutoModelForCausalLM.from_pretrained(args.base_model_path,torch_dtype=dtype,device_map=device,attn_implementation="eager",low_cpu_mem_usage=True)
            model.eval()
            correct = 0
            total = 0
            pbar = tqdm(dataset.iterrows(), total=len(dataset), desc="HF sst2")
            for _, row in pbar:
                sentence = row['sentence']
                label = str(row['label']).strip()
                prompt = get_sst2_prompt(sentence)
                st = time.time()
                inputs = tokenizer(prompt, return_tensors="pt").input_ids
                input_ids = torch.as_tensor(inputs).to(device)
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
                pbar.set_postfix(acc=f"{correct/total:.2%}")
            print("HF Model's accuracy on sst2 is: ",correct/total)        
              
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default="")
    parser.add_argument("--draft-model-path", type=str, default="")
    parser.add_argument("--dataset", type=str, default="mt_bench")
    parser.add_argument("--task", type=str,choices=['speed', 'accuracy'], default="speed")
    parser.add_argument("--predictor-path", type=str, default="")
    parser.add_argument("--model-size", type=str,choices=['7B'],default="7B")
    parser.add_argument("--pred-thresholds", type=float,default=0.5)
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum new tokens to generate (default: 256)")

    args = parser.parse_args()
    main(args)
