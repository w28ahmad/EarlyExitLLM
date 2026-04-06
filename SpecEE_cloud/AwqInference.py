import argparse
import torch
from fastchat.model import get_conversation_template
import os
import time
from tqdm import trange
import json
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import pyarrow.parquet as pq
from awq import AutoAWQForCausalLM
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
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

def main(args):
    model_name_or_path = args.base_model_path
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path,device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    model.eval()
    ls = ['mt_bench','sum','qa','alpaca','gsm8k','humaneval']
    speed = []
    for f in ls:
        question_list = load_questions('./benchmark/'+f+'/question.jsonl',begin=0,end=80)
        output_ids_tot = 0
        st = time.time()
        for i in trange(len(question_list)):
            torch.cuda.empty_cache()
            message = question_list[i]['turns'][0]
            conv = get_conversation_template("llama-2-chat")  
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            conv.append_message(conv.roles[0], message)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            input_ids=tokenizer([prompt]).input_ids
            seqlen = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            output_ids=model.generate(input_ids,max_new_tokens=256)
            output_ids_tot += len(output_ids[0]) - seqlen
            output=tokenizer.decode(output_ids[0])
        ed = time.time()
        speed.append(output_ids_tot/(ed-st))
    data = dict(zip(ls, speed))
    json_file_path = 'raw_awq.json'
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, default='')
    args = parser.parse_args()
    main(args)