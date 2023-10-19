# -*- coding = utf-8 -*-
from rank_bm25 import BM25Okapi
import numpy as np
import json
from datasets import load_dataset
from datasets import load_from_disk
from tqdm import tqdm
# # glue
import datasets

dataset = load_from_disk('./rawdata/sick')

train_examples = dataset["train"]
dev_examples = dataset["validation"]

corpus=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['sentence_A']
    s2=train_examples[i]['sentence_B']
    sent=s1+s2
    corpus.append(sent)
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

stage_1_res = {}

for i in range(len(dev_examples)):
    s1 = train_examples[i]['sentence_A']
    s2 = train_examples[i]['sentence_B']
    sent = s1 + s2
    query=sent
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:100] #取前多少个
    # stage_1_res[i]=top_n.tolist()
    stage_1_res[i]={}
    stage_1_res[i]['index']=top_n.tolist()
    stage_1_res[i]['scores'] = [scores[tt] for tt in top_n]

with open('sick_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res, ensure_ascii=False))

print('sick finish')

dataset = load_dataset("glue","cola")
train_examples = dataset["train"]
dev_examples = dataset["validation"]

corpus=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['sentence']
    sent=s1
    corpus.append(sent)
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

stage_1_res = {}
stage_1_res_100={}
progressbar = tqdm(range(len(dev_examples)))
for i, batch in enumerate(dev_examples):
    s1=dev_examples[i]['sentence']
    query=s1
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n_100=np.argsort(scores)[::-1][:100] #取前多少个

    stage_1_res_100[i] = {}
    stage_1_res_100[i]['index'] = top_n_100.tolist()
    stage_1_res_100[i]['scores'] = [scores[tt] for tt in top_n_100]

    progressbar.update(1)

with open('cola_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res_100, ensure_ascii=False))
#
print('cola finish')

dataset = load_dataset("tweet_eval", "stance_feminist")  # 女权主义者
# dataset = load_from_disk('./rawdata/tweet_eval_stance_feminist')

# 0:none 1:反对 2:支持
train_examples = dataset["train"]
dev_examples = dataset["validation"]

corpus=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['text']
    corpus.append(s1)
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

stage_1_res = {}

for i in range(len(dev_examples)):
    s1=dev_examples[i]['text']
    query=s1
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:100] #取前多少个
    # stage_1_res[i]=top_n.tolist()
    stage_1_res[i]={}
    stage_1_res[i]['index']=top_n.tolist()
    stage_1_res[i]['scores'] = [scores[tt] for tt in top_n]
with open('tweet_eval_stance_feminist_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res, ensure_ascii=False))

print('stance_feminist finish')

dataset = load_dataset("tweet_eval", "stance_hillary")  # 女权主义者
# dataset = load_from_disk('./rawdata/tweet_eval_stance_hillary')
# 0:none 1:反对 2:支持
train_examples = dataset["train"]
dev_examples = dataset["validation"]
# dev_examples[0]:
# {'text': 'You can have everything in this life, but as long as you are not pleasing #God, there will always be a void in your heart. #SemST',
#  'label': 1}

corpus=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['text']
    corpus.append(s1)
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

stage_1_res = {}

for i in range(len(dev_examples)):
    s1=dev_examples[i]['text']
    query=s1
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:100] #取前多少个
    # stage_1_res[i]=top_n.tolist()
    stage_1_res[i]={}
    stage_1_res[i]['index']=top_n.tolist()
    stage_1_res[i]['scores'] = [scores[tt] for tt in top_n]
with open('tweet_eval_stance_hillary_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res, ensure_ascii=False))

print('tweet_eval_stance_hillary finish')

#
# # ethos
dataset = load_dataset('ethos', 'multilabel', split='train')
# dataset = load_from_disk('./rawdata/ethos')

# ethos_race[15]
# {'text': 'what a disgusting man and disgusting religion. He admits that muslim buy all their wifes ! .pff...This is why all muslims are poor...islam keeps them stupid !',
#  'race': 0}
train_size=346
validation_size=87

# ethos_race:  3个数据集都用的同一个text，只是最后测demostration的label的时候不一样
# 通过seed=42保证划分数据一致
dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size,seed=42)
train_examples, dev_examples = dataset['train'], dataset['test']
print('dev_examples[3]:{}'.format(dev_examples[3]))
print('dev_examples[17]:{}'.format(dev_examples[17]))
corpus=[]
for i in range(len(train_examples)):
    s1=train_examples[i]['text']
    corpus.append(s1)
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

stage_1_res = {}

for i in range(len(dev_examples)):
    s1=dev_examples[i]['text']
    query=s1
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:100] #取前多少个
    # stage_1_res[i]=top_n.tolist()
    stage_1_res[i]={}
    stage_1_res[i]['index']=top_n.tolist()
    stage_1_res[i]['scores'] = [scores[tt] for tt in top_n]
with open('ethos_stage13_new.json', mode='w', encoding='utf-8') as f:
    f.write(json.dumps(stage_1_res, ensure_ascii=False))
# #
print('ethos finish')
# #




