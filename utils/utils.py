# utils.py
# Contains random utility functions that are useful for multiple tasks
import os
import glob
from copy import deepcopy
import torch
import numpy as np
from transformers import EncoderDecoderModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertLMHeadModel
from models.charformer_bert import BertWithGBSTEncDec, BertWithGBSTEncoder
from models.lee_bert import BertWithLeeFull
from models.twostep import TwoStep
from models.other_charformers import BertWithMaskedGBSTEncDec, BertWithPaddedGBSTEncDec

def padding_collate_fn(batch, max_len=1024):
    """ 
        Pads each list with zeros and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    padded_batch = {}
    for key in batch[0]:
        largest = min(max_len, max([len(b[key]) for b in batch]))
        padded_batch[key] = torch.zeros((len(batch), largest), dtype=torch.long)
        if key == "labels":
            padded_batch[key] -= 100
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, :key_len] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def tokenize(examples, tokenizer):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'labels':[]}

    for example in examples['translation']:
        src = example[tokenizer.langs[0]]
        tgt = example[tokenizer.langs[1]]

        src_tok = tokenizer.encode(src)
        encoded['input_ids'] += [src_tok]
        
        tgt_tok = tokenizer.encode(tgt)
        encoded['decoder_input_ids'] += [[tokenizer.bos]*tokenizer.bos_buffer + tgt_tok[:-tokenizer.bos_buffer]]
        encoded['labels'] += [tgt_tok]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]
    return encoded


def get_reload_path(args):
    reload_id = args.reload_id if args.reload_id else os.environ.get('SLURM_JOB_ID')
    reload_path = "dumped/%s/%s/%s/" % (args.task, args.model_type, reload_id)
    return reload_path

def get_checkpoint_path(reload_path):
    checkpoint_paths = glob.glob(reload_path + "checkpoint-*")    
    # always save best and last, so best will always have the lowest number
    checkpoint_step_nums = [int(x.split("-")[-1]) for x in checkpoint_paths]
    best_checkpoint = np.argmin(checkpoint_step_nums)
    best_checkpoint_path = checkpoint_paths[best_checkpoint]
    return best_checkpoint_path

def load_model(pretrained, args):
    config = BertConfig(
        vocab_size=args.tok.vocab_size, 
        hidden_size=args.emb_dim, 
        num_hidden_layers=6 if not args.tiny else 4, 
        num_attention_heads=8, 
        intermediate_size=args.emb_dim*4,
        max_position_embeddings=1024,
        ) # Transformer Base

    if args.reload_path:
        pretrained = get_checkpoint_path(args.reload_path)

    if args.model_type == "gbst":
        if args.reload_path:
            model = BertWithGBSTEncoder.from_pretrained(pretrained, args.freeze, config, ds_factor=args.ds_factor)
        else:
            model = BertWithGBSTEncoder(pretrained, args.freeze, bert_config=config, ds_factor=args.ds_factor)
        return model
        
    if args.model_type in "full/full_causal".split("/"):
        causal = args.model_type == "full_causal"
        if args.reload_path:
            model = BertWithGBSTEncDec.from_pretrained(pretrained, args.freeze, config, causal=causal, ds_factor=args.ds_factor)
        else:
            model = BertWithGBSTEncDec(pretrained, args.freeze, bert_config=config, causal=causal, ds_factor=args.ds_factor)
        return model

    if args.model_type == "masked_full":
        if args.reload_path:
            model = BertWithMaskedGBSTEncDec.from_pretrained(pretrained, args.freeze, config, ds_factor=args.ds_factor)
        else:
            model = BertWithMaskedGBSTEncDec(pretrained, args.freeze, bert_config=config, ds_factor=args.ds_factor)
        return model

    if args.model_type == "padded_full":
        if args.reload_path:
            model = BertWithPaddedGBSTEncDec.from_pretrained(pretrained, args.freeze, config, ds_factor=args.ds_factor)
        else:
            model = BertWithPaddedGBSTEncDec(pretrained, args.freeze, bert_config=config, ds_factor=args.ds_factor)
        return model

    if args.model_type == "full_lee":
        if args.reload_path:
            model = BertWithLeeFull.from_pretrained(pretrained, args.freeze, config, causal=True, ds_factor=args.ds_factor)
        else:
            model = BertWithLeeFull(pretrained, args.freeze, bert_config=config, causal=True, ds_factor=args.ds_factor)
        return model

    if args.model_type == "twostep" or args.model_type == "twostep_word":
        if args.reload_path:
            model = TwoStep.from_pretrained(pretrained, args.freeze, config)
        else:
            model = TwoStep(pretrained, args.freeze, bert_config=config)
        return model
       
    if args.model_type == "word" or args.model_type == "char":
        if args.reload_path:
            model = EncoderDecoderModel.from_pretrained(pretrained)
        else:
            enc_config = deepcopy(config)
            enc_model = BertModel(enc_config)
            
            dec_config = deepcopy(config)
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_model = BertLMHeadModel(dec_config)

            model = EncoderDecoderModel(encoder=enc_model, decoder=dec_model)
        return model
    
    assert False, "%s is not a valid model type." % args.model_type