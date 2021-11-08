# pretrain.py
import datasets
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import ByT5Tokenizer # newest version has a bug
from byt5_tokenizer import ByT5Tokenizer
from datasets import load_dataset, load_metric
from functools import partial
from types import MethodType
from pretrain.pretrain_eval import evaluation_loop
from utils.utils import padding_collate_fn, load_model
from utils.mask_t5 import mask_span_sentinel

import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser("Preprocessing dataset for pretraining")
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): {t5 = word, byt5 = char}. \
                    To be added: {charformer = gbst, fullchar = full}" )
parser.add_argument("--save_path", type=str, help="Path of dataset saving.")


LANG1 = "en"
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 258

def tokenize(examples, poisson_lambda, sent_range):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'labels':[]}

    for example in examples['translation']:
        x_tok = torch.LongTensor(tok.encode(example[LANG1]))
        src_tok, tgt_tok = mask_span_sentinel(x_tok, poisson_lambda=poisson_lambda, sentinel_range=sent_range)

        # print(example[LANG1])
        # print(tok.decode(src_tok), tok.decode(tgt_tok))
        
        encoded['input_ids'] += [src_tok.tolist()]        
        encoded['decoder_input_ids'] += [tgt_tok[:-1].tolist()]
        encoded['labels'] += [tgt_tok[1:].tolist()]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

    return encoded

if __name__ == "__main__":
    args = parser.parse_args()
    args.model_type = args.model_type.lower()
    char = True
    if args.model_type == "t5" or args.model_type == "word":
        char = False
        
    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)

    dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')

    tokenize = partial(tokenize, sent_range=((258, 358) if char else (32000, 32100)))
    tokenize = partial(tokenize, poisson_lambda=20.0 if char else 3.5, sent_range=((158, 258) if char else (32000, 32100)))
    dataset = dataset.map(tokenize, batched=True, num_proc=10)

    dataset.save_to_disk(args.save_path)

    # dataset = datasets.load_from_disk(args.save_path)
    # print(dataset['train'][0])