import argparse
import sys
import os
from EE_model_awq_kernel import EEModel
import torch
from fastchat.model import get_conversation_template
import os
import time
from tqdm import trange
import json
from model_llama_ee import MLP
from typing import Optional
import pandas as pd
import pyarrow.parquet as pq
from torch import nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
    
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
    model = EEModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.draft_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="eager"
    )
    model.eval()
    ls = ['mt_bench','sum','qa','alpaca','gsm8k','humaneval']
    speed = []
    for f in ls:
        question_list = load_questions('./benchmark/'+f+'/question.jsonl',begin=0,end=80)
        exit_layer_id_list=[]
        output_ids_tot = 0
        st = time.time()
        torch.cuda.empty_cache()
        for i in trange(len(question_list)):
            torch.cuda.empty_cache()
            message = question_list[i]['turns'][0]
            conv = get_conversation_template("llama-2-chat")  
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "    
            input_ids=model.tokenizer([prompt]).input_ids
            seqlen = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            output_ids=model(input_ids,max_new_tokens=256,exit_layer_id_list=exit_layer_id_list)
            output_ids_tot += len(output_ids[0]) - seqlen
            output=model.tokenizer.decode(output_ids[0])
        ed = time.time()
        speed.append(output_ids_tot/(ed-st))
    data = dict(zip(ls, speed))
    json_file_path = 'specee_awq.json'
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default='')
    parser.add_argument("--draft-model-path", type=str, default="")
    args = parser.parse_args()
    main(args)
