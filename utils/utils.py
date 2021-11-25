# utils.py
# Contains random utility functions that are useful for multiple tasks
from copy import deepcopy
import torch
from transformers import EncoderDecoderModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import BertModel, BertLMHeadModel
from models.charformer_bert import BertWithGBSTFull, BertWithGBSTEncoder

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


def load_model(pretrained, args):
    config = BertConfig(
        vocab_size=args.vocab_size, 
        hidden_size=args.emb_dim, 
        num_hidden_layers=6 if not args.tiny else 4, 
        num_attention_heads=8, 
        intermediate_size=args.emb_dim*4,
        max_position_embeddings=1024,
        ) # Transformer Base

    if args.model_type == "gbst":
        if args.reload_path:
            model = BertWithGBSTEncoder.from_pretrained(pretrained, args.freeze, config)
        else:
            model = BertWithGBSTEncoder(pretrained, args.freeze, bert_config=config)
        return model
        
    if args.model_type in "full/full_causal".split("/"):
        if args.reload_path:
            raise NotImplementedError

        causal = args.model_type == "full_causal"
        if args.reload_path:
            model = BertWithGBSTFull.from_pretrained(pretrained, args.freeze, config, causal=causal)
        else:
            model = BertWithGBSTFull(pretrained, args.freeze, bert_config=config, causal=causal)
        return model
    
    if args.model_type == "stacked_bert":
        from models.bert_stacked import BertStackingModel, BertStackingLMHeadModel
        if args.reload_path:
            raise NotImplementedError

        enc_config = deepcopy(config)
        enc_model = BertStackingModel(enc_config)

        dec_config = deepcopy(config)
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        dec_model = BertStackingLMHeadModel(dec_config)

        model = EncoderDecoderModel(encoder=enc_model, decoder=dec_model)
        return model
    
    if args.model_type == "word" or args.model_type == "char":
        if args.reload_path:
            raise NotImplementedError
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