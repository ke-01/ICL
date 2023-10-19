#  将demostration修改为每个test有自己的实例

import logging
import os
import random
import time
import csv
import json
import torch
from torch import nn
# os.environ["cuda_visible_devices"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["cuda_visible_devices"] = "0,1"
device = torch.device('cuda:'+'0') if torch.cuda.is_available() else torch.device('cpu')
# 2 0 1 3
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from datasets import load_dataset, load_metric, DatasetDict,load_from_disk
import datasets
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer


dtype_dict = {
    'float64':torch.float64,
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}

@hydra.main(version_base=None, config_path="conf", config_name="config3")
def main(config: DictConfig) -> None:
    start_time = time.time()
    # for debugging purpose
    torch.set_printoptions(profile="full")

    device = torch.device(config['ds_configs']['device'])
    dtype = dtype_dict[config['ds_configs']['dtype']]
    device = torch.device('cuda:' + '0') if torch.cuda.is_available() else torch.device('cpu')
    print(device)
    print(dtype)

    # Set default seed 100 just in case! (e.g. model bias?)
    set_seed(100)
    random.seed(100)

    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO)

    dataset = DatasetDict()
    # train_path = config['train_path']
    # datasets_train=[]
    #
    # dataset['train'] = load_dataset('json', data_files=train_path)['train']
    # print(dataset['train'])
    test_path = config['test_path']
    dataset['test'] = load_dataset('json', data_files=test_path)['train']

    def train_preprocess_function(example):
        # Tokenize the texts
        input_sentence = config['experiments']["template"]
        input_sentence = input_sentence.replace('[BOL]', '')
        input_sentence = input_sentence.replace('[S1]', example['sentence1'])
        if "sentence2" in example:
            input_sentence = input_sentence.replace('[S2]', example['sentence2'])

        label = example['label']

        input_sentence = input_sentence.replace('[Label]', config["experiments"]["verbalizer"][int(label)])
        example['input_sentence'] = input_sentence
        return example

    def test_preprocess_function(example, idx):
        # Tokenize the texts
        input_sentence = config['experiments']["template"]
        input_sentence = input_sentence.replace('[S1]', example['sentence1'])
        if "sentence2" in example:
            input_sentence = input_sentence.replace('[S2]', example['sentence2'])

        return {"input_sentence": input_sentence, "idx": idx}

    # train_dataset = dataset['train'].map(
    #     train_preprocess_function,
    #     desc="Preprocessing dataset...",
    #     fn_kwargs={'acc': config['demo_accuracy']}
    # )
    test_dataset = dataset['test'].map(
        test_preprocess_function,
        desc="Preprocessing dataset...",
        with_indices=True,
    )
    # datasets_train = []
    data_type=config['experiments']['task']
    llm_model=config['models']['name']
    print(data_type)
    print(llm_model)
    top_n=config['top_n']

	# for ours
	if data_type=='disability':
        with open('../chatglm/top16_threshold4_v2_' + str(threshold) + '_' + 'ethos' + '_stage2_' + str(
                llm_model) + '_torch32.json', 'r',
                  encoding='utf8')as f:
            stage_2_index = json.load(f)
    else:
        with open('../chatglm/top16_threshold4_v2_'+str(threshold)+'_'+ str(data_type) + '_stage2_' + str(llm_model) + '_torch32.json', 'r',
                  encoding='utf8')as f:
            stage_2_index = json.load(f)

	# for bm25
    if data_type=='disability':
        with open('../chatglm/' + 'ethos'+ '_stage13_new.json', 'r',
                  encoding='utf8')as f:
            stage_2_index = json.load(f)
    else:
        with open('../chatglm/'+str(data_type) + '_stage13_new.json', 'r',
                  encoding='utf8')as f:
            stage_2_index = json.load(f)
	
    for key, values in stage_2_index.items():
        stage_2_index[key] = stage_2_index[key]['index'][:top_n]


    if  data_type == 'cola':
        if data_type == 'cola':
            dataset = load_dataset("glue", "cola")
        train_examples = dataset["train"]
        dev_examples = dataset["validation"]


        train_dataset = []
        for index in range(len(dev_examples)):
            stage_2_data = [train_examples[id] for id in stage_2_index[str(index)]]

            selected_samples = {}
            selected_samples['input_sentence'] = []
            for ii in range(len(stage_2_data)):
                ss1 = stage_2_data[ii]['sentence']
                label = stage_2_data[ii]['label']
                # sent = s1 + s2
                sample_dict = dict()
                sample_dict['sentence1'] = ss1
                # if sentence2:
                #     sample_dict['sentence2'] = sentence2
                sample_dict['label'] = label
                example = train_preprocess_function(sample_dict)
                selected_samples['input_sentence'].append(example['input_sentence'])
            train_dataset.append(selected_samples)
        print('len train:{}'.format(len(train_dataset)))
    elif data_type == 'sick':
        dataset = load_from_disk('../chatglm/rawdata/sick')
        train_examples = dataset["train"]
        dev_examples = dataset["validation"]
        # with open('../chatglm/threshold_v2_' + str(data_type) + '_stage2_' + str(llm_model) + '_torch32.json', 'r',
        #           encoding='utf8')as f:
        #     stage_2_index = json.load(f)

        train_dataset = []
        for index in range(len(dev_examples)):
            # 得到当前query对应的第2阶段的数据集
            stage_2_data = [train_examples[id] for id in stage_2_index[str(index)]]

            selected_samples = {}
            selected_samples['input_sentence'] = []
            for ii in range(len(stage_2_data)):
                ss1 = stage_2_data[ii]['sentence_A']
                ss2 = stage_2_data[ii]['sentence_B']
                label = stage_2_data[ii]['label']
                # sent = s1 + s2
                sample_dict = dict()
                sample_dict['sentence1'] = ss1
                sample_dict['sentence2'] = ss2
                sample_dict['label'] = label
                example = train_preprocess_function(sample_dict)
                selected_samples['input_sentence'].append(example['input_sentence'])
            train_dataset.append(selected_samples)
        print('len train:{}'.format(len(train_dataset)))
    elif (data_type == 'tweet_eval_stance_feminist' or
      data_type == 'tweet_eval_stance_hillary' ):

        if data_type=='tweet_eval_stance_feminist':
            dataset = load_from_disk("../chatglm/rawdata/tweet_eval_stance_feminist")
        elif data_type=='tweet_eval_stance_hillary':
            dataset = load_from_disk("../chatglm/rawdata/tweet_eval_stance_hillary")


        train_examples = dataset["train"]
        dev_examples = dataset["validation"]


        train_dataset = []
        for index in range(len(dev_examples)):
            stage_2_data = [train_examples[id] for id in stage_2_index[str(index)]]

            selected_samples = {}
            selected_samples['input_sentence'] = []
            for ii in range(len(stage_2_data)):
                ss1 = stage_2_data[ii]['text']
                label = stage_2_data[ii]['label']
                # sent = s1 + s2
                sample_dict = dict()
                sample_dict['sentence1'] = ss1
                # sample_dict['sentence2'] = ss2
                sample_dict['label'] = label
                example = train_preprocess_function(sample_dict)
                selected_samples['input_sentence'].append(example['input_sentence'])
            train_dataset.append(selected_samples)
        print('len train:{}'.format(len(train_dataset)))

    elif data_type == 'disability':
        # dataset = load_dataset("super_glue", "cb")
        dataset = load_from_disk('../chatglm/rawdata/ethos')
        train_size = 346
        validation_size = 87
        dataset = dataset.train_test_split(train_size=train_size, test_size=validation_size,seed=42)
        train_examples, dev_examples = dataset['train'], dataset['test']


        train_dataset = []
        for index in range(len(dev_examples)):
            stage_2_data = [train_examples[id] for id in stage_2_index[str(index)]]

            selected_samples = {}
            selected_samples['input_sentence'] = []
            for ii in range(len(stage_2_data)):
                ss1 = stage_2_data[ii]['text']
                label = stage_2_data[ii]['disability']
                # sent = s1 + s2
                sample_dict = dict()
                sample_dict['sentence1'] = ss1
                # sample_dict['sentence2'] = ss2
                sample_dict['label'] = label
                example = train_preprocess_function(sample_dict)
                selected_samples['input_sentence'].append(example['input_sentence'])
            train_dataset.append(selected_samples)
        print('len train:{}'.format(len(train_dataset)))

    demonstrations=[]
    for i in range(len(test_dataset)):
        dt = [input_sentence for input_sentence in train_dataset[i]['input_sentence']]
        dt = config['experiments']['demo_sep'].join(dt)
        demonstrations.append(dt)
    print('len de:{}'.format(len(demonstrations)))

    # print(demonstrations)

    # Set tokenizer
    # print('token:{}'.format(config['models']["model_name_or_path"]))

    if llm_model=='gpt2' or llm_model=='gpt2-xl'  or llm_model=='gpt2-medium'or llm_model=='gpt2-large':
        tokenizer = AutoTokenizer.from_pretrained(config['models']["model_name_or_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    bol_token = config['models']['bol_token']
    if config['models']['add_bol_to_special_token']:
        tokenizer.add_special_tokens({'additional_special_tokens': [bol_token]})

    bol_token_id = tokenizer.encode(bol_token)[-1]
    print('bol_id:{}'.format(bol_token_id))

    # Load model
    logger.info(f'Start loading model {config["models"]["model_name_or_path"]}')
    model_loading_start = time.time()
    if llm_model == 'gpt2' or llm_model == 'gpt2-xl' or llm_model=='gpt2-medium'or llm_model=='gpt2-large':
        model = AutoModelForCausalLM.from_pretrained(config["models"]["model_name_or_path"], torch_dtype=torch.float32).to(device)
    print(config["models"]["model_name_or_path"])

    model.eval()
    model_loading_end = time.time()
    logger.info(f'Total time for loading model : {model_loading_end - model_loading_start} sec.')

    batch_size = config['ds_configs']['train_micro_batch_size_per_gpu']

    # Evaluate! 
    logger.info("***** Zero/Few-shot Evaluation *****")
    logger.info(f"  TASK                                = {config['experiments']['task']}")
    logger.info(f"  Num TRAIN examples                  = {len(train_dataset)}")
    logger.info(f"  Num TEST  examples                  = {len(test_dataset)}")
    logger.info(f"  Random Seed                         = {config['seed']}")
    logger.info(f"  Demo accuracy                       = {config['demo_accuracy']}")
    logger.info(f"  Inference Model                     = {config['models']['model_name_or_path']}")
    logger.info(f'=== in-context samples ===\n{demonstrations[0]}\n=====================')

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)  # 未打乱
    metric = load_metric('custom_metric.py')

    cnt=0
    progressbar = tqdm(range(len(dataloader)))
    for step, batch in enumerate(dataloader):
        sentence_inputs = []
        dummy_inputs = []
        current_batch_size = len(batch['input_sentence'])
        # 对batch内的每一个数据
        for i in range(current_batch_size):
            for label_idx, label_token in config['experiments']['verbalizer'].items():
                sentence_with_label = batch['input_sentence'][i].replace('[Label]', label_token)
                sentence_with_label = demonstrations[cnt] + config['experiments']['demo_sep'] + sentence_with_label
                dummy_inputs.append(sentence_with_label.replace('[BOL]', bol_token))
                sentence_inputs.append(sentence_with_label.replace('[BOL]', ''))
            cnt += 1


        inputs = tokenizer(sentence_inputs, padding=True, return_tensors='pt').to(device)
        # inputs = tokenizer(sentence_inputs, padding=True, return_tensors='pt').cuda()

        labels = inputs['input_ids']
        token_length = inputs['input_ids'].size(-1)

        dummy_inputs = tokenizer(dummy_inputs, padding=True, return_tensors='pt').to(device)
        # dummy_inputs = tokenizer(dummy_inputs, padding=True, return_tensors='pt').cuda()


        bol_indices = (dummy_inputs['input_ids'] == bol_token_id).nonzero()[:,1]
        if llm_model=='gpt-j':
            label_masks = [torch.cat((torch.zeros(idx), torch.ones(1), torch.zeros(token_length - idx - 1))) for idx in
                           bol_indices]
            label_masks = torch.stack(label_masks).to(device)
            label_masks = label_masks  # gpt-j需要改
        else:

            label_masks = [torch.cat((torch.zeros(idx), torch.ones(token_length - idx))) for idx in bol_indices]
            label_masks = torch.stack(label_masks).to(device)
            label_masks = label_masks * inputs['attention_mask']
            # label_masks = torch.stack(label_masks).cuda()

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        label_masks = label_masks[..., 1:].contiguous()

        #### to cpu float32 ####
        labels = labels.cpu().detach()
        label_masks = label_masks.cpu().detach()
        logits = logits.cpu().detach()
        logits = logits.to(torch.float32)

        losses = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        losses = losses.view(logits.size(0), -1)
        losses = losses * label_masks
        losses = torch.sum(losses, axis=1)
        losses = losses.view(current_batch_size, -1)
        prediction = torch.argmin(losses, dim=1)

        progressbar.update(1)

        references=[{"idx": batch['idx'][i], "label": batch['label'][i], "probs": losses[i]} for i in range(current_batch_size)]
        metric.add_batch(predictions=prediction, references=references)

    result = metric.compute()

    logger.info(f"  ACCURACY                     = {result['accuracy']}")
    logger.info(f"  F1                           = {result['f1']}")
    with open(os.path.join(config['output_path'], f"acc-{config['demo_accuracy']}-seed-{config['seed']}_predictions.jsonl"), 'a') as prediction_file:
        for l, p, prob in zip(result['labels'], result['predictions'], result['probs']):
            json.dump({"label": l, "prediction": p, "prob": prob}, prediction_file)
            prediction_file.write('\n')

    end_time = time.time()
    logger.info(f'Total runtime : {end_time - start_time} sec.')

if __name__ == "__main__":
    main()
