# nmt.py
import os
import transformers
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer_callback import PrinterCallback
from byt5_tokenizer_new import ByT5Tokenizer
import datasets
from datasets import load_dataset
from functools import partial
from types import MethodType
from utils.utils import padding_collate_fn, load_model
from utils.metrics import compute_bleu
from utils.logging_utils import load_logger, LogFlushCallback

import torch
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger = load_logger(logger)

parser = argparse.ArgumentParser("Fine-tuning NMT")
# Model args
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): {word, char, gbst, full, full_causal}" )
parser.add_argument("--reload_path", type=str, help="Path of model to reload.")
parser.add_argument("--no_pretrain", action="store_true", help="Don't use a pretrained model. \
                    Only applies if reload_path is not provided.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 4 encoder layers and 2 decoder layers.")
parser.add_argument("--freeze", type=str, default="", help="Freeze transformer portion")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")

# Data args
parser.add_argument("--langs", type=str, default="de,en", help="Languages used, comma-separated.")
parser.add_argument("--spm_model", type=str, default="", help="Path of sentencepiece model.")
parser.add_argument("--data_lim", type=int, default=0, help="Number of sentences to use, or 0 to disable.")
parser.add_argument("--bos_buffer", type=int, default=1, help="Number of BOS tokens to pad on the beginning. \
                    Already included for full GBST model. Only relevant for specific experimental models.")
# Training args
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--grad_acc", type=int, default=1, help="Accumulate gradients.")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")

def tokenize(examples, args):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'labels':[]}

    for example in examples['translation']:
        src = example[args.langs[0]]
        tgt = example[args.langs[1]]

        src_tok = args.tok.encode(src)
        encoded['input_ids'] += [src_tok]
        
        tgt_tok = args.tok.encode(tgt)
        encoded['decoder_input_ids'] += [[args.bos_token_id]*args.bos_buffer + tgt_tok[:-args.bos_buffer]]
        encoded['labels'] += [tgt_tok]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]
    return encoded

if __name__ == "__main__":
    args = parser.parse_args()
    args.model_type = args.model_type.lower()
    args.langs = args.langs.split(",")
    logger.info(args)
    datasets.logging.set_verbosity_error()

    char = True
    if args.model_type == "t5" or args.model_type == "word":
        char = False

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10

        
    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)
    if args.spm_model:
        assert args.model_type == "word"
        tok = T5Tokenizer(args.spm_model, model_max_length=512)

    args.tok = tok
    args.vocab_size = tok.vocab_size
    args.eos_token_id = 1
    args.bos_token_id = tok.vocab_size - 1

    pretrained_model = pretrained
    if args.reload_path:
        assert not args.no_pretrain
        pretrained_model = args.reload_path
    model = load_model(pretrained_model, args)

    model.config.bos_token_id = args.bos_token_id

    # dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')
    dataset = load_dataset('dataloading/iwslt2017_dataset.py', 'iwslt2017-de-en')

    if args.debug and args.eval_only:
        dataset = dataset.filter(lambda example, idx: idx < 10, with_indices=True)

    if args.data_lim:
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    tokenize = partial(tokenize, args=args)
    dataset = dataset.map(tokenize, batched=True, num_proc=10)

    training_args = Seq2SeqTrainingArguments("dumped/nmt/%s/%s/" % (args.model_type, os.environ.get('SLURM_JOB_ID')),
        evaluation_strategy="steps",
        save_strategy="steps",
        generation_max_length=1024 if char else 256,
        predict_with_generate=True,
        group_by_length=True,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        log_level="info",
        save_steps=args.save_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        warmup_steps=4000,
        learning_rate=5e-4,
        label_smoothing_factor=0.1,
        load_best_model_at_end=True,
        disable_tqdm=not args.debug)

    early_stopping = EarlyStoppingCallback(early_stopping_patience=10)
    print_cb = LogFlushCallback(logger)
    compute_metrics = partial(compute_bleu, args=args)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=padding_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping, print_cb]
        )

    trainer.pop_callback(PrinterCallback)

    # from models.charformer import compute_loss
    # trainer.compute_loss = MethodType(compute_loss, trainer)
    # trainer.evaluation_loop = MethodType(evaluation_loop, trainer)
    # from models.rainbow import prediction_step
    # trainer.prediction_step = MethodType(prediction_step, trainer)

    if not args.eval_only:
        trainer.train()
    trainer.evaluate(dataset['test'])