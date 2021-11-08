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

parser = argparse.ArgumentParser("Pretraining")
parser.add_argument("--model_type", type=str, help="The model type to use. \
                    Currently supported (case insensitive): {t5 = word, byt5 = char}. \
                    To be added: {charformer = gbst, fullchar = full}" )
parser.add_argument("--data_path", type=str, help="Path of preprocessed dataset. \
                    If not provided, will preprocess from scratch.")
parser.add_argument("--reload_path", type=str, help="Path of model to reload.")
parser.add_argument("--no_pretrain", action="store_true", help="Don't use a pretrained model. \
                    Only applies if reload_path is not provided.")
parser.add_argument("--debug", action="store_true", help="Activates debug mode, \
                    which shortens steps until evaluation, and enables tqdm.")
parser.add_argument("--tiny", action="store_true", help="Use a tiny model, \
                    with 4 encoder layers and 2 decoder layers.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--freeze", action="store_true", help="Freeze transformer portion")


# GBST params
# parser.add_argument("--gbst_word", action="store_true", help="Use T5 with gbst")


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

bleu = load_metric('sacrebleu')
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # print(predictions.shape, labels.shape)
    # print(logits[0].shape, logits[1].shape)
    predictions = np.argmax(logits[0], axis=-1)
    char_preds = []
    char_labels = []
    for b in range(predictions.shape[0]):
        # pred = predictions[b][predictions[b]!=-100]
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

        char_preds.append(tok.decode(pred, skip_special_tokens=True))
        char_labels.append([tok.decode(lab, skip_special_tokens=True)])
        # char_labels.append([from_char(lab)])

    print("\npred:", char_preds[0:10])
    print("\nlabel:", char_labels[0:10])

    bleu_score = {"bleu": bleu.compute(predictions=char_preds, references=char_labels)['score']}
    print(bleu_score, flush=True)
    return bleu_score


acc = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # print(logits[0].shape, logits[1].shape)

    predictions = logits[0] #np.argmax(logits[0], axis=-1)

    predictions = predictions.flatten()
    labels = labels.flatten()

    mask = labels != -100

    labels = labels[mask]
    predictions = predictions[mask]

    return acc.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    args = parser.parse_args()
    args.model_type = args.model_type.lower()
    char = True
    if args.model_type == "t5" or args.model_type == "word":
        char = False

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 100
        # args.save_steps = 100

    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)

    pretrained_model = pretrained
    if args.reload_path:
        assert not args.no_pretrain
        pretrained_model = args.reload_path
    model, config = load_model(pretrained_model, args)

    model.config.bos_token_id = BOS_TOKEN_ID

    if args.data_path:
        dataset = datasets.load_from_disk(args.data_path)
    else:
        dataset = load_dataset('dataloading/translation_dataset.py', 'de-en')
        tokenize = partial(tokenize, poisson_lambda=20.0 if char else 3.5, sent_range=((158, 258) if char else (32000, 32100)))
        dataset = dataset.map(tokenize, batched=True)

    training_args = Seq2SeqTrainingArguments("dumped/pretrain/%s" % args.model_type,
        evaluation_strategy="steps",
        generation_max_length=1024,
        # predict_with_generate=True,
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        learning_rate=1e-3,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        disable_tqdm=not args.debug)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['test'],
        data_collator=padding_collate_fn,
        compute_metrics=compute_metrics
        )

    trainer.evaluation_loop = MethodType(evaluation_loop, trainer)

    trainer.train()