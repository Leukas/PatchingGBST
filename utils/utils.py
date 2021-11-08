# utils.py
# Contains random utility functions that are useful for multiple tasks
import torch
from transformers import T5Config, T5ForConditionalGeneration, EncoderDecoderModel, EncoderDecoderConfig
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bert.configuration_bert import BertConfig
# from models.t5_hf import T5ForConditionalGeneration
from models.charformer import T5WithGBSTEncoder, T5WithGBSTFull

from models.rainbow import BartForConditionalGeneration, T5Rainbow, RBD, BertModel

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

    # print(padded_batch)

    return padded_batch


def load_logger(logger):
    import time, datetime
    import logging
    class LogFormatter():
        def __init__(self):
            self.start_time = time.time()

        def format(self, msg):
            time_passed = round(msg.created - self.start_time)
            prefix = "[%s - %s]" % (
                time.strftime('%x %X'),
                datetime.timedelta(seconds=time_passed)
            )
            msg_text = msg.getMessage()
            return "%s %s" % (prefix, msg_text) if msg_text else ''

    logger.setLevel(logging.INFO)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(LogFormatter())
    logger.addHandler(log_handler)
    return logger

def load_model_old(pretrained, args):
    tiny_kwargs = {
        "num_layers": 4,
        "num_decoder_layers": 2
    }
    base_kwargs = {
        "num_layers": 6,
        "num_decoder_layers": 6
    }
    if args.model_type == "gbst":
        if args.reload_path:
            model = T5WithGBSTEncoder.from_pretrained(pretrained, args.freeze)
        else:
            model = T5WithGBSTEncoder(pretrained, args.freeze)
        config = model.t5.config
        return model, config
    elif args.model_type == "gbst_word":
        if args.reload_path:
            model = T5WithGBSTEncoder.from_pretrained('t5-small', args.freeze)
        else:
            model = T5WithGBSTEncoder('t5-small', args.freeze)
        config = model.t5.config
        return model, config
    elif args.model_type == "full":
        model = T5WithGBSTFull(pretrained, args.freeze)
        config = model.t5.config
        return model, config


    if not args.no_pretrain:
        if args.tiny:
            model = T5ForConditionalGeneration.from_pretrained(pretrained, **tiny_kwargs)
        else:
            model = T5ForConditionalGeneration.from_pretrained(pretrained)
    else:
        if args.tiny:
            config = T5Config.from_pretrained(pretrained, **tiny_kwargs)
        else:
            config = T5Config.from_pretrained(pretrained)
        model = T5ForConditionalGeneration(config)

    if args.freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True

        if args.freeze == "half":
            for i in range(6):
                for param in model.encoder.block[i].parameters():
                    param.requires_grad = True


    model.train()
    config = model.config # set config to match pretrained
    return model, config

def load_model(pretrained, args):
    tiny_kwargs = {
        "num_layers": 4,
        "num_decoder_layers": 4,
    } if args.tiny else {}

    if args.model_type == "gbst":
        if args.reload_path:
            raise NotImplementedError

        config = T5Config(vocab_size=384, d_model=args.emb_dim, **tiny_kwargs) # same size as Transformer Base
        model = T5WithGBSTEncoder(pretrained, args.freeze, t5_config=config)
        return model, config
    if args.model_type == "full":
        if args.reload_path:
            raise NotImplementedError

        config = T5Config(vocab_size=384, d_model=args.emb_dim, **tiny_kwargs) # same size as Transformer Base
        if args.reload_path:
            model = T5WithGBSTFull.from_pretrained(pretrained, args.freeze, config)
            config = model.config
        model = T5WithGBSTFull(pretrained, args.freeze, t5_config=config)
        return model, config

        # if args.reload_path:
        #     model = T5WithGBSTEncoder.from_pretrained(pretrained, args.freeze)
        # else:
    
    if args.model_type == "rainbow":
        if args.reload_path:
            raise NotImplementedError
        enc_config = BertConfig(vocab_size=384, d_model=args.emb_dim, max_position_embeddings=1024) # same size as Transformer Base
        # config = BartConfig(vocab_size=384) # same size as Transformer Base
        # model = BartForConditionalGeneration(config)
        dec_config = BertConfig(vocab_size=384, d_model=args.emb_dim, max_position_embeddings=1024) # same size as Transformer Base
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        dec_config.num_hidden_layers = 11
        enc = BertModel(enc_config)
        dec = RBD(dec_config)
        model = EncoderDecoderModel(encoder=enc, decoder=dec)
    if args.model_type == "bert":
        if args.reload_path:
            raise NotImplementedError
        from transformers.models.bert.modeling_bert import BertModel
        enc_config = BertConfig(vocab_size=384, d_model=args.emb_dim, max_position_embeddings=1024, attention_probs_dropout_prob=0.5) # same size as Transformer Base
        # config = BartConfig(vocab_size=384) # same size as Transformer Base
        # model = BartForConditionalGeneration(config)
        dec_config = BertConfig(vocab_size=384, d_model=args.emb_dim, max_position_embeddings=1024, attention_probs_dropout_prob=0.5) # same size as Transformer Base
        dec_config.is_decoder = True
        dec_config.add_cross_attention = True
        # enc = BertModel(enc_config)
        # dec = BertModel(dec_config)
        # model = EncoderDecoderModel(encoder=enc, decoder=dec)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
        model = EncoderDecoderModel(config)
        # model = BartForConditionalGeneration(config)
        return model, model.config
    if args.model_type == "rainbow_t5":
        if args.reload_path:
            config = T5Config(vocab_size=384, d_model=args.emb_dim) # same size as Transformer Base
            model = T5Rainbow.from_pretrained(pretrained)
        else:
            config = T5Config(vocab_size=384, d_model=args.emb_dim) # same size as Transformer Base
            model = T5Rainbow(config)
        return model, model.config
    


    if args.model_type == "word":
        if args.reload_path:
            config = T5Config(**tiny_kwargs) # same size as Transformer Base
            model = T5ForConditionalGeneration.from_pretrained(pretrained)
        else:
            config = T5Config(**tiny_kwargs) # same size as Transformer Base
            model = T5ForConditionalGeneration(config)
        return model, model.config
    if args.reload_path:
        config = T5Config(vocab_size=384, **tiny_kwargs) # same size as Transformer Base
        model = T5ForConditionalGeneration.from_pretrained(pretrained)
    else:
        config = T5Config(vocab_size=384, **tiny_kwargs) # same size as Transformer Base
        model = T5ForConditionalGeneration(config)

    # if args.freeze:
    #     for param in model.parameters():
    #         param.requires_grad = False
    #     for param in model.lm_head.parameters():
    #         param.requires_grad = True

    #     if args.freeze == "half":
    #         for i in range(6):
    #             for param in model.encoder.block[i].parameters():
    #                 param.requires_grad = True


    model.train()
    config = model.config # set config to match pretrained
    return model, config





    # model = T5ForConditionalGeneration.from_pretrained(pretrained,
    #         num_layers=4,
    #         num_decoder_layers=2)
    model = T5ForConditionalGeneration.from_pretrained(pretrained)
    # config = T5Config.from_pretrained(pretrained)
    # config.num_layers = 4
    # config.num_decoder_layers = 2
    # model = T5ForConditionalGeneration(config)
    model.train()