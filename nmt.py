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
from utils.utils import get_reload_path, tokenize, padding_collate_fn, load_model
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
                    Currently supported (case insensitive): \
                    {word, char, gbst, full, full_causal, masked_full, padded_full, full_lee, twostep, twostep_word}" )
parser.add_argument("--reload_id", type=str, help="Job ID of model to reload.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 4 layers.")
parser.add_argument("--freeze", type=str, default="", help="Freeze transformer portion")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")
parser.add_argument("--ds_factor", type=int, default=4, help="Downsampling factor for gbst and full")
# Data args
parser.add_argument("--langs", type=str, default="de,en", help="Languages used, comma-separated.")
parser.add_argument("--spm_model", type=str, default="", help="Path of sentencepiece model. Only relevant for word models.")
parser.add_argument("--data_lim", type=int, default=0, help="Limit number of sentences to use, or 0 to disable.")
parser.add_argument("--bos_buffer", type=int, default=1, help="Number of BOS tokens to pad on the beginning. \
                    Already included for full GBST model. Only relevant for specific experimental models.")
# Training args
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--grad_acc", type=int, default=1, help="Accumulate gradients.")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=5000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")

if __name__ == "__main__":
    args = parser.parse_args()
    args.task = "nmt"
    args.model_type = args.model_type.lower()
    args.langs = args.langs.split(",")
    logger.info(args)
    datasets.logging.set_verbosity_error()

    char = True
    if args.model_type == "twostep_word" or args.model_type == "word":
        char = False

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10

        
    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)
    if args.spm_model:
        assert "word" in args.model_type
        tok = T5Tokenizer(args.spm_model, model_max_length=512)

    args.tok = tok
    args.tok.bos = tok.vocab_size - 1
    args.tok.bos_buffer = args.bos_buffer
    args.tok.langs = args.langs

    pretrained_model = pretrained
    args.reload_path = get_reload_path(args) if args.reload_id else None 

    model = load_model(pretrained_model, args)

    model.config.bos_token_id = args.tok.bos

    # dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')
    dataset = load_dataset('dataloading/iwslt2017_dataset.py', 'iwslt2017-%s-%s' % tuple(args.langs))

    # if args.debug:
    #     dataset['validation'] = dataset['validation'].filter(lambda example, idx: idx < 10, with_indices=True)

    # filter out sentence pairs where english side is longer than 256 chars (allows for larger batch sizes)
    dataset['train'] = dataset['train'].filter(lambda example: len(example['translation']['en']) < 256, num_proc=10)
    print("After filtering:", dataset['train'])

    if args.data_lim: # limit amount of training data
        dataset['train'] = dataset['train'].filter(lambda example, idx: idx < args.data_lim, with_indices=True)

    tokenize = partial(tokenize, tokenizer=tok)
    dataset = dataset.map(tokenize, batched=True, num_proc=10)

    args.save_path = args.reload_path if args.reload_id else get_reload_path(args)

    training_args = Seq2SeqTrainingArguments(args.save_path,
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
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        warmup_steps=4000,
        learning_rate=args.lr,
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
    trainer._max_length = 1024 if char else 256
    trainer._num_beams = 1

    trainer.pop_callback(PrinterCallback)

    # from models.charformer import compute_loss
    # trainer.compute_loss = MethodType(compute_loss, trainer)
    from nmt.nmt_eval import evaluation_loop
    trainer.evaluation_loop = MethodType(evaluation_loop, trainer)
    # trainer.evaluate = MethodType(evaluate, trainer)
    # trainer.callback_handler.on_evaluate = MethodType(on_evaluate, trainer.callback_handler)

    if not args.eval_only:
        trainer.train()
    
    args.eval_only = True # for writing test set to file
    trainer.evaluate(dataset['test'])