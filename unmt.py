# unmt.py
import os
import torch
import numpy as np
import argparse
import sys 
import logging
import transformers
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback, PrinterCallback
# from transformers import ByT5Tokenizer # newest version has a bug
from byt5_tokenizer_new import ByT5Tokenizer
import datasets
from datasets import load_dataset, load_metric
from functools import partial
from types import MethodType

from nmt.nmt_eval import evaluation_loop
from unmt.unmt_training_step import compute_loss
from utils.utils import padding_collate_fn, load_model, load_logger

logger = load_logger(logging.getLogger(__name__))

parser = argparse.ArgumentParser("Fine-tuning NMT")
# Model args
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): {t5 = word, byt5 = char}. \
                    To be added: {charformer = gbst, fullchar = full}" )
parser.add_argument("--reload_path", type=str, help="Path of model to reload.")
parser.add_argument("--no_pretrain", action="store_true", help="Don't use a pretrained model. \
                    Only applies if reload_path is not provided.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 4 encoder layers and 2 decoder layers.")
parser.add_argument("--freeze", type=str, default="", help="Freeze transformer portion")
parser.add_argument("--emb_dim", type=int, default=512, help="Size of embedding dimension.")

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

LANG1 = "de"
LANG2 = "en"
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 258
LANG1_TOKEN_ID = 259
LANG2_TOKEN_ID = 260
    
def tokenize_nmt(examples):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'labels':[]}

    for example in examples['translation']:
        src = example[LANG1]
        tgt = example[LANG2]

        src_tok = tok.encode(src)
        src_tok = [LANG1_TOKEN_ID] + src_tok
        encoded['input_ids'] += [src_tok]
        
        tgt_tok = tok.encode(tgt)
        encoded['decoder_input_ids'] += [[LANG2_TOKEN_ID] + tgt_tok[:-1]]
        encoded['labels'] += [tgt_tok]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

    return encoded

def tokenize_unmt(examples):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'decoder_attention_mask':[]}

    for example in examples['translation']:
        src = example[LANG1]
        tgt = example[LANG2]

        src_tok = tok.encode(src)
        src_tok = [LANG1_TOKEN_ID] + src_tok
        encoded['input_ids'] += [src_tok]

        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]
        
        tgt_tok = tok.encode(tgt)
        tgt_tok = [LANG2_TOKEN_ID] + tgt_tok
        encoded['decoder_input_ids'] += [tgt_tok]

        attention_mask = torch.ones(len(tgt_tok), dtype=torch.long).tolist()
        encoded['decoder_attention_mask'] += [attention_mask]

    return encoded
    
bleu = load_metric('sacrebleu')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    char_preds = []
    char_labels = []
    for b in range(predictions.shape[0]):
        pred = predictions[b]
        if (pred==EOS_TOKEN_ID).sum() > 0:
            eos_idx = np.argmax(pred==EOS_TOKEN_ID)
            pred = pred[:eos_idx]

        lab = labels[b]
        if (pred==EOS_TOKEN_ID).sum() > 0:
            eos_idx = np.argmax(lab==EOS_TOKEN_ID)
            lab = lab[:eos_idx]
        else:
            lab = lab[lab!=-100]

        if b < 5:
            print("P:", pred)
            print("L:", lab)
        char_preds.append(tok.decode(pred, skip_special_tokens=True))
        char_labels.append([tok.decode(lab, skip_special_tokens=True)])

    print("\npred:", char_preds[0:5])
    print("\nlabel:", char_labels[0:5])

    bleu_score = {"bleu": bleu.compute(predictions=char_preds, references=char_labels)['score']}

    sys.stdout.flush()
    return bleu_score

class LogFlushCallback(TrainerCallback):
    """ Like printer callback, but with logger and flushes the logs every call """
    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            logger.info(logs)
            sys.stdout.flush()

if __name__ == "__main__":
    args = parser.parse_args()
    args.model_type = args.model_type.lower()
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

    pretrained_model = None
    if args.reload_path:
        assert not args.no_pretrain
        pretrained_model = args.reload_path
    model, config = load_model(pretrained_model, args)

    model.config.bos_token_id = LANG2_TOKEN_ID

    dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')
    dataset['train'] = dataset['train'].map(tokenize_unmt, batched=True, num_proc=10)
    dataset['validation'] = dataset['validation'].map(tokenize_nmt, batched=True, num_proc=10)
    dataset['test'] = dataset['test'].map(tokenize_nmt, batched=True, num_proc=10)

    if not char:
        dataset = dataset.filter(lambda example: len(example['input_ids']) < 512, num_proc=10)
        dataset = dataset.filter(lambda example: len(example['decoder_input_ids']) < 512, num_proc=10)

    if args.debug and args.eval_only:
        dataset = dataset.filter(lambda example, idx: idx < 10, with_indices=True)


    training_args = Seq2SeqTrainingArguments("dumped/unmt/%s/%s/" % (args.model_type, os.environ.get('SLURM_JOB_ID')),
        evaluation_strategy="steps",
        save_strategy="steps",
        generation_max_length=1024,
        predict_with_generate=True,
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
    print_cb = LogFlushCallback()
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

    trainer.evaluation_loop = MethodType(evaluation_loop, trainer)
    trainer.compute_loss = MethodType(compute_loss, trainer)

    trainer.LANG1_TOKEN_ID = LANG1_TOKEN_ID
    trainer.LANG2_TOKEN_ID = LANG2_TOKEN_ID
    trainer._max_length=1024

    if not args.eval_only:
        trainer.train()
    trainer.evaluate(dataset['test'])