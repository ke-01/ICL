from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    set_seed,
    GPT2Model
)
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer,GPT2Model
import torch
import time
import transformers

import json
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import os
import os
# from scipy.linalg import pinv2
from datasets import load_dataset,load_from_disk
import datasets
import argparse
import math
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device('cuda:'+'3') if torch.cuda.is_available() else torch.device('cpu')
print(device)

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--data_type', type=str, default="cola", help='dataset choice')
parser.add_argument('--llm_model', type=str, default="gpt2", help='llm_model choice')
parser.add_argument('--threshold', type=float, default="0.5", help='llm_model choice')

args = parser.parse_args()

data_type=args.data_type
llm_model=args.llm_model
threshold=args.threshold
print('data_type:{}'.format(data_type))
print('llm_model:{}'.format(llm_model))
print('threshold:{}'.format(threshold))

with open(str(data_type)+'_stage13_new.json', 'r', encoding='utf8')as f:
    stage_1_index = json.load(f)

if data_type=='sick':
    dataset = load_from_disk('./rawdata/sick')
    train_examples = dataset["train"]
    dev_examples = dataset["validation"]
elif data_type=='cola':
    dataset = load_dataset("glue", "cola")
    train_examples = dataset["train"]
    dev_examples = dataset["validation"]
elif data_type=='tweet_eval_stance_feminist':
    dataset = load_dataset("tweet_eval", "stance_feminist")
    # dataset = load_from_disk("./rawdata/tweet_eval_stance_feminist")
    train_examples = dataset["train"]
    dev_examples = dataset["validation"]
elif data_type=='tweet_eval_stance_hillary':
    dataset = load_dataset("tweet_eval", "stance_hillary")
    # dataset = load_from_disk("./rawdata/tweet_eval_stance_hillary")
    train_examples = dataset["train"]
    dev_examples = dataset["validation"]

elif data_type=='ethos':
    dataset = load_from_disk('./rawdata/ethos')
    train_size=346
    validation_size=87
    dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size,seed=42)
    train_examples, dev_examples = dataset['train'], dataset['test']
    print('ethos:dev_examples[3]:{}'.format(dev_examples[3]))
    print('dev_examples[17]:{}'.format(dev_examples[17]))



if llm_model=='gpt2':
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32).to(device)
elif llm_model=='gpt2-medium':
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium', torch_dtype=torch.float32).to(device)
elif llm_model=='gpt2-large':
    tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = AutoModelForCausalLM.from_pretrained('gpt2-large', torch_dtype=torch.float32).to(device)
elif llm_model=='gpt2-xl':
    tokenizer = AutoTokenizer.from_pretrained('/new_disk2/kepu_zhang/pretrain_model/gpt2-xl')
    model = AutoModelForCausalLM.from_pretrained('/new_disk2/kepu_zhang/pretrain_model/gpt2-xl', torch_dtype=torch.float32).to(device)

model = model.eval()
# for name, _ in model.named_parameters():
#     print(name)
history = []


def cal_Gp_gpt2(sent,max_len):
    input_tensor = tokenizer(sent, return_tensors="pt").to(device)
    L=input_tensor['input_ids'].shape[1]
    while L<max_len: #complate
        nums_l=int(max_len/L)+1
        sent=sent*nums_l
        input_tensor = tokenizer(sent, return_tensors="pt").to(device)
        L = input_tensor['input_ids'].shape[1]

    mask=input_tensor['attention_mask']
    ctx_ids=input_tensor['input_ids']
    # ethos
    if L>1000:
        L=1000
        ctx_ids=ctx_ids[:,:1000]
        mask=mask[:,:1000]
    with torch.no_grad():
        next_kv = model(
            input_ids=ctx_ids,
            attention_mask=mask,
            past_key_values=None,
            use_cache=True,
        ).past_key_values  # kv @ (old_ctx + new_ctx)

    layer_k, layer_v=next_kv[0]
    num_heads = layer_k.shape[1]
    attn_head_size = layer_k.shape[3]
    tensor_v = layer_v.permute(0, 2, 1, 3).contiguous()  # 1 11 12 64

    new_shape = tensor_v.size()[:-2] + (num_heads * attn_head_size,)
    tt_v = tensor_v.view(new_shape)  # 1,11,768
    Gp=tt_v
    Gp = Gp.view(Gp.shape[1], Gp.shape[2])
    Gp=Gp[-max_len:,:]
    return Gp


def get_50_percent_lengths(lengths_list):
    sorted_lengths = sorted(lengths_list)

    index_percent = int(len(sorted_lengths) *threshold)

    lengths_percent = sorted_lengths[:index_percent]

    return lengths_percent[-1]

def cal_influc_fnorm(Hess_es,Gp_z):

    influence_score = -torch.matmul(Hess_es, Gp_z) #8,768
    norm = torch.norm(influence_score, 'fro')/math.sqrt(influence_score.shape[0]*influence_score.shape[1]) #
    return norm
def min_max_normalize(lst):
    if all(element == 0 for element in lst)==True:
        return lst
    min_val = min(lst)
    max_val = max(lst)
    if max_val - min_val==0:
        print('all same!!')
        print(lst)
        return lst
    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst


our_stage2_res={}
stage_2_res_16={}
if data_type == 'cola' :
    progressbar = tqdm(range(len(dev_examples)))
    for index, batch in enumerate(dev_examples):
        s1 = dev_examples[index]['sentence']
        query = s1
        stage_1_data = [train_examples[id] for id in stage_1_index[str(index)]['index']]

        lengths = []
        for ii in range(len(stage_1_data)):
            sent = stage_1_data[ii]['sentence']
            input_tensor = tokenizer(sent, return_tensors="pt")
            L = input_tensor['input_ids'].shape[1]
            lengths.append(L)

        percent_50_lengths = get_50_percent_lengths(lengths)
        max_len = percent_50_lengths

        influence_score_list = []
        Hess_es = 0
        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['sentence']
            sent = ss1
            Gp = cal_Gp_gpt2(sent, max_len)
            Hess_es += (1 / len(stage_1_data)) * torch.matmul(Gp, Gp.transpose(0, 1))
        Hess_es = torch.pinverse(Hess_es)

        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['sentence']
            sent = ss1
            Gp_z=cal_Gp_gpt2(sent,max_len)
            norm = cal_influc_fnorm(Hess_es, Gp_z)
            influence_score_list.append(norm.item())
        influence_score_list = min_max_normalize(influence_score_list)

        our_stage2_res[index] = {}
        our_stage2_res[index]['index'] = stage_1_index[str(index)]['index']
        our_stage2_res[index]['scores'] = influence_score_list
        bm25_score_list=stage_1_index[str(index)]['scores']
        bm25_score_list= min_max_normalize(bm25_score_list)
        for ii in range(len(stage_1_data)):
            influence_score_list[ii]+=bm25_score_list[ii]

        top_n = np.argsort(influence_score_list)[::-1]
        top_n = top_n.tolist()
        top_n=[stage_1_index[str(index)]['index'][id] for id in top_n]

        stage_2_t_16 = top_n[:16]
        assert len(stage_2_t_16) == 16
        stage_2_res_16[index] = stage_2_t_16
        # top_ns[index] = top_n
        progressbar.update(1)
    with open('our_l1_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(our_stage2_res, ensure_ascii=False))
    with open('top16_threshold4_v2_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(stage_2_res_16, ensure_ascii=False))
elif (data_type == 'tweet_eval_stance_feminist' or data_type == 'tweet_eval_stance_hillary' or data_type == 'ethos'):
    progressbar = tqdm(range(len(dev_examples)))
    for index, batch in enumerate(dev_examples):
        s1 = dev_examples[index]['text']
        query = s1
        stage_1_data = [train_examples[id] for id in stage_1_index[str(index)]['index']]

        max_len=0
        lengths=[]
        for ii in range(len(stage_1_data)):
            sent = stage_1_data[ii]['text']
            input_tensor = tokenizer(sent, return_tensors="pt")
            L = input_tensor['input_ids'].shape[1]
            lengths.append(L)

        percent_50_lengths = get_50_percent_lengths(lengths)
        max_len=percent_50_lengths
        Hess_es = 0
        influence_score_list = []
        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['text']
            sent = ss1
            Gp = cal_Gp_gpt2(sent, max_len)
            tm=torch.matmul(Gp, Gp.transpose(0, 1))
            Hess_es += (1 / len(stage_1_data)) * tm
        Hess_es = torch.pinverse(Hess_es)  #

        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['text']
            sent = ss1
            Gp_z=cal_Gp_gpt2(sent,max_len)
            norm = cal_influc_fnorm(Hess_es, Gp_z)
            influence_score_list.append(norm.item())

        influence_score_list = min_max_normalize(influence_score_list)
        our_stage2_res[index] = {}
        our_stage2_res[index]['index'] = stage_1_index[str(index)]['index']
        our_stage2_res[index]['scores'] = influence_score_list

        bm25_score_list = stage_1_index[str(index)]['scores']
        bm25_score_list = min_max_normalize(bm25_score_list)
        for ii in range(len(stage_1_data)):
            influence_score_list[ii] += bm25_score_list[ii]#

        influence_score_list = min_max_normalize(influence_score_list)
        top_n = np.argsort(influence_score_list)[::-1]
        top_n = top_n.tolist()
        top_n=[stage_1_index[str(index)]['index'][id] for id in top_n]

        stage_2_t_16 = top_n[:16]
        assert len(stage_2_t_16) == 16
        stage_2_res_16[index] = stage_2_t_16
        progressbar.update(1)
    with open('our_l1_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(our_stage2_res, ensure_ascii=False))
    with open('top16_threshold4_v2_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(stage_2_res_16, ensure_ascii=False))
elif data_type == 'sick':
    progressbar = tqdm(range(len(dev_examples)))
    for index, batch in enumerate(dev_examples):
        s1 = dev_examples[index]['sentence_A']
        s2 = dev_examples[index]['sentence_B']
        query = s1 + s2
        stage_1_data = [train_examples[id] for id in stage_1_index[str(index)]['index']]

        lengths = []
        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['sentence_A']
            ss2 = stage_1_data[ii]['sentence_B']
            sent = ss1 + ss2
            input_tensor = tokenizer(sent, return_tensors="pt")
            L = input_tensor['input_ids'].shape[1]
            lengths.append(L)

        percent_50_lengths = get_50_percent_lengths(lengths)
        max_len = percent_50_lengths

        influence_score_list = []
        Hess_es = 0
        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['sentence_A']
            ss2 = stage_1_data[ii]['sentence_B']
            sent = ss1 + ss2
            Gp = cal_Gp_gpt2(sent, max_len)

            Hess_es += (1 / len(stage_1_data)) * torch.matmul(Gp, Gp.transpose(0, 1))
        Hess_es = torch.pinverse(Hess_es)

        for ii in range(len(stage_1_data)):
            ss1 = stage_1_data[ii]['sentence_A']
            ss2 = stage_1_data[ii]['sentence_B']
            sent = ss1 + ss2
            Gp_z=cal_Gp_gpt2(sent,max_len)
            norm = cal_influc_fnorm(Hess_es, Gp_z)
            influence_score_list.append(norm.item())

        influence_score_list = min_max_normalize(influence_score_list)

        our_stage2_res[index] = {}
        our_stage2_res[index]['index'] = stage_1_index[str(index)]['index']
        our_stage2_res[index]['scores'] = influence_score_list
        bm25_score_list = stage_1_index[str(index)]['scores']
        bm25_score_list = min_max_normalize(bm25_score_list)
        for ii in range(len(stage_1_data)):
            influence_score_list[ii]+=bm25_score_list[ii]

        top_n = np.argsort(influence_score_list)[::-1]
        top_n = top_n.tolist()
        top_n=[stage_1_index[str(index)]['index'][id] for id in top_n]

        stage_2_t_16 = top_n[:16]
        assert len(stage_2_t_16) == 16
        stage_2_res_16[index] = stage_2_t_16

        progressbar.update(1)
    with open('our_l1_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(our_stage2_res, ensure_ascii=False))
    with open('top16_threshold4_v2_' + str(threshold) + '_' + str(data_type) + '_stage2_' + str(
            llm_model) + '_torch32.json',
              mode='w',
              encoding='utf-8') as f:
        f.write(json.dumps(stage_2_res_16, ensure_ascii=False))



