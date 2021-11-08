# glue.py
import os
from transformers import T5Tokenizer, TrainingArguments, Trainer
# from transformers import ByT5Tokenizer # newest version has a bug
from byt5_tokenizer import ByT5Tokenizer
from datasets import load_dataset, load_metric
from functools import partial
import torch
import numpy as np
import argparse
from utils.utils import load_model, padding_collate_fn

parser = argparse.ArgumentParser("Pretraining")
parser.add_argument("--model_type", type=str, default="char", help="The model type to use. \
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
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging.")
parser.add_argument("--freeze", type=str, default="", help="Freeze transformer portion")

BOS_TOKEN_ID = 2

# def tokenize(examples):
#     return tok(examples["sentence"])#, padding="longest")


# def tokenize(examples):
#     encoded = {'input_ids':[], 'attention_mask':[]}

#     for example in examples['sentence']:
#         input_ids = (torch.tensor([list(example.encode("utf-8"))]) + 3).tolist()[0]
#         # input_ids = tok.encode(example)
#         # assert (torch.tensor(input_ids[:-1]) == torch.tensor(input_ids2)).all(), str(input_ids2) + " ||| " + str(input_ids[:-1])
#         attention_mask = torch.ones(len(input_ids)).tolist()
#         encoded['input_ids'] += [input_ids]
#         encoded['attention_mask'] += [attention_mask]

#     return encoded


def tokenize(examples):
    encoded = {'input_ids':[], 'attention_mask':[], 'decoder_input_ids':[], 'labels':[]}

    for sentence in examples['sentence']:
        src_tok = tok.encode(sentence)
        encoded['input_ids'] += [src_tok]
        
        attention_mask = torch.ones(len(src_tok), dtype=torch.long).tolist()
        encoded['attention_mask'] += [attention_mask]

    for label in examples['label']:
        encoded['decoder_input_ids'] += [[BOS_TOKEN_ID]]
        encoded['labels'] += [[label]]

    # print(encoded)

    return encoded


# def tokenize_s2(examples):
#     return tok(examples["sentence2"], padding="max_length")

# def _padding_collate_fn(batch, pad_token):
#     # print(batch)
#     # largest_sample = len(batch[0])
#     largest_sample = max([len(b['input_ids']) for b in batch])
#     padded_batch = torch.full((len(batch), largest_sample), pad_token, dtype=torch.long)
#     padded_attn_mask = torch.zeros_like(padded_batch)
#     for i, sample in enumerate(batch):
#         padded_batch[i, :len(sample['input_ids'])] = torch.LongTensor(sample['input_ids'])
#         padded_attn_mask[i, :len(sample['attention_mask'])] = torch.LongTensor(sample['attention_mask'])
#         # batch[i]['attention_mask'] = padded_attn_mask[i]
#         # batch[i]['input_ids'] = padded_batch[i]

#     dec_ids = torch.zeros(padded_batch.size(0), 1).long()
#     labels = torch.LongTensor([b['label'] for b in batch]).unsqueeze(1)
#     return {'input_ids': padded_batch, 'attention_mask': padded_attn_mask, 'decoder_input_ids': dec_ids, 'labels': labels}
    
# def padding_collate_fn(pad_token=0):
#     return partial(_padding_collate_fn, pad_token=pad_token)

acc = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # print(logits[0].shape, logits[1].shape)
    predictions = np.argmax(logits[0], axis=-1)
    print("", flush=True)
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

    # pretrained = 'google/byt5-small'
    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)

    pretrained_model = pretrained
    if args.reload_path:
        assert not args.no_pretrain
        pretrained_model = args.reload_path
    model, config = load_model(pretrained_model, args)

    dataset = load_dataset('glue', 'sst2')
    dataset = dataset.map(tokenize, batched=True, remove_columns=["label"])

    print(dataset)

    training_args = TrainingArguments("dumped/pretrain/%s/%s/" % (args.model_type, os.environ.get('SLURM_JOB_ID')),
        evaluation_strategy="steps",
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        disable_tqdm=not args.debug)

    # print(training_args.device)

    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['validation'],
        data_collator=padding_collate_fn,
        compute_metrics=compute_metrics
        )

    trainer.train()