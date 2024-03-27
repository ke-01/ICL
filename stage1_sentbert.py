
# -*- coding = utf-8 -*-
# from rank_bm25 import BM25Okapi
import torch
import numpy as np
import json
from datasets import load_dataset
from datasets import load_from_disk
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer as SBert
import datasets
from tqdm import tqdm

device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

model = SBert('./paraphrase-multilingual-MiniLM-L12-v2')
model.to(device)

# glue
dataset = load_dataset("glue","cola")
train_examples = dataset["train"]
dev_examples = dataset["validation"]
stage_1_res = {}

dev_set=[]
for i in range(len(dev_examples)):
    s1=dev_examples[i]['sentence']
    sent=s1
    dev_set.append(sent)

train_sent=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['sentence']
    sent = s1
    train_sent.append(sent)

print('begin calcute:')
q_embeddings = model.encode(dev_set)
d_embeddings = model.encode(train_sent)
cosine_X_Y = cos_sim(q_embeddings, d_embeddings)  # tensor
print('cos finish')

stage_1_res_100={}
a1, top_n_100 = torch.topk(cosine_X_Y, 100, dim=1, largest=True, sorted=True, out=None)
top_n_100 = top_n_100.tolist()
a1 = a1.tolist()
progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100[i]
    stage_1_res_100[i]['scores'] = a1[i]
    progressbar.update(1)

with open('cola_sentbert_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))

print('cola finish')


dataset = load_from_disk('./rawdata/sick')
train_examples = dataset["train"]
dev_examples = dataset["validation"]

dev_set=[]
for i in range(len(dev_examples)):
    s1 = train_examples[i]['sentence_A']
    s2 = train_examples[i]['sentence_B']
    sent = s1 + s2
    dev_set.append(sent)

train_sent=[]
for i in range(len(train_examples)):
    s1 = train_examples[i]['sentence_A']
    s2 = train_examples[i]['sentence_B']
    sent = s1 + s2
    train_sent.append(sent)

q_embeddings = model.encode(dev_set)
d_embeddings = model.encode(train_sent)
cosine_X_Y = cos_sim(q_embeddings, d_embeddings)  # tensor

stage_1_res_100={}
a1, top_n_100 = torch.topk(cosine_X_Y, 100, dim=1, largest=True, sorted=True, out=None)
top_n_100=top_n_100.tolist()
a1=a1.tolist()

progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100[i]
    stage_1_res_100[i]['scores'] = a1[i]
    progressbar.update(1)

with open('sick_sentbert_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))

print('sick finish')

dataset = load_from_disk('./rawdata/tweet_eval_stance_hillary')
train_examples = dataset["train"]
dev_examples = dataset["validation"]
stage_1_res = {}
dev_set=[]
for i in range(len(dev_examples)):
    query=dev_examples[i]['text']
    dev_set.append(query)

train_sent=[]
for ii in range(len(train_examples)):
    s1 = train_examples[ii]['text']
    train_sent.append(s1)

q_embeddings = model.encode(dev_set)
d_embeddings = model.encode(train_sent)
cosine_X_Y = cos_sim(q_embeddings, d_embeddings)  # tensor

stage_1_res_100={}
a1, top_n_100 = torch.topk(cosine_X_Y, 100, dim=1, largest=True, sorted=True, out=None)
top_n_100 = top_n_100.tolist()
a1 = a1.tolist()
progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100[i]
    stage_1_res_100[i]['scores'] = a1[i]
    progressbar.update(1)
with open('tweet_eval_stance_hillary_sentbert_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))

print('stance_hillary finish')


dataset = load_from_disk('./rawdata/tweet_eval_stance_feminist')

train_examples = dataset["train"]
dev_examples = dataset["validation"]
stage_1_res = {}
dev_set=[]
for i in range(len(dev_examples)):
    query=dev_examples[i]['text']
    dev_set.append(query)

train_sent=[]
for ii in range(len(train_examples)):
    s1 = train_examples[ii]['text']
    train_sent.append(s1)

q_embeddings = model.encode(dev_set)
d_embeddings = model.encode(train_sent)
cosine_X_Y = cos_sim(q_embeddings, d_embeddings)  # tensor

stage_1_res_100={}
a1, top_n_100 = torch.topk(cosine_X_Y, 100, dim=1, largest=True, sorted=True, out=None)
top_n_100 = top_n_100.tolist()
a1 = a1.tolist()

progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100[i]
    stage_1_res_100[i]['scores'] = a1[i]
    progressbar.update(1)

with open('tweet_eval_stance_feminist_sentbert_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))

print('stance_feminist finish')


dataset = load_from_disk('./rawdata/ethos')

train_size=346
validation_size=87

dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size,seed=42)
train_examples, dev_examples = dataset['train'], dataset['test']

dev_set=[]
stage_1_res = {}
for i in range(len(dev_examples)):
    query=dev_examples[i]['text']
    dev_set.append(query)

train_sent=[]
for ii in range(len(train_examples)):
    s1 = train_examples[ii]['text']
    train_sent.append(s1)

q_embeddings = model.encode(dev_set)
d_embeddings = model.encode(train_sent)
cosine_X_Y = cos_sim(q_embeddings, d_embeddings)  # tensor

stage_1_res_100={}
a1, top_n_100 = torch.topk(cosine_X_Y, 100, dim=1, largest=True, sorted=True, out=None)
top_n_100=top_n_100.tolist()
a1=a1.tolist()
progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100[i]
    stage_1_res_100[i]['scores'] = a1[i]
    progressbar.update(1)

with open('ethos_sentbert_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))
#
print('ethos finish')