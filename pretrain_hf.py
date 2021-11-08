# pretrain.py
import datasets
from transformers import T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import ByT5Tokenizer # newest version has a bug
from byt5_tokenizer_new import ByT5Tokenizer
from datasets import load_dataset, load_metric
from functools import partial
from types import MethodType
from pretrain.pretrain_eval import evaluation_loop
from utils.utils import load_model_old, padding_collate_fn, load_model
from utils.mask_t5 import mask_span_sentinel
from utils.mask_t5_temp import FlaxDataCollatorForT5MLM, compute_input_and_target_lengths

import os
import torch
import numpy as np
import argparse

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
parser.add_argument("--grad_acc", type=int, default=1, help="Number of steps to accumulate gradients. \
                    So the true batch size will be batch_size * grad_acc")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluation.")
parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving.")
parser.add_argument("--logging_steps", type=int, default=200, help="Number of steps between logging.")
parser.add_argument("--freeze", type=str, default="", help="Freeze transformer portion")
parser.add_argument("--eval_only", action="store_true", help="Only evaluate.")


# GBST params
# parser.add_argument("--gbst_word", action="store_true", help="Use T5 with gbst")


LANG1 = "de"
LANG2 = "en"
EOS_TOKEN_ID = 1
BOS_TOKEN_ID = 258

def tokenize(examples):
    encoded = {'input_ids':[]}#, 'attention_mask':[]}

    for lang in [LANG1, LANG2]:
        for example in examples['translation']:
            x_tok = tok.encode(example[lang])
            # print(x_tok)
            # src_tok, tgt_tok = mask_span_sentinel(x_tok, poisson_lambda=poisson_lambda, sentinel_range=sent_range)

            # print(example[LANG1])
            # print(tok.decode(src_tok), tok.decode(tgt_tok))
            encoded['input_ids'] += [x_tok]        
        # attention_mask = torch.ones(len(x_tok), dtype=torch.long).tolist()
        # encoded['attention_mask'] += [attention_mask]

    return encoded



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

    print("predt:", predictions[:100])
    print("labelt:", labels[:100])

    # print("pred:", "".join(tok.convert_ids_to_tokens(predictions[:100])))
    # print("label:", "".join(tok.convert_ids_to_tokens(labels[:100])))
    # NOTE: convert_ids_to_tokens does not work correctly with characters like ö

    print("pred:", tok.decode(predictions[:100]))
    print("label:", tok.decode(labels[:100]))

    print("Acc:", flush=True)
    return acc.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    args = parser.parse_args()
    args.model_type = args.model_type.lower()
    datasets.logging.set_verbosity_error()
    char = True
    if args.model_type == "t5" or args.model_type == "word":
        char = False

    if args.debug:
        args.logging_steps = 1
        args.eval_steps = 10
        # args.save_steps = 100

    pretrained = 'google/byt5-small' if char else 't5-small'
    tok = ByT5Tokenizer.from_pretrained(pretrained) if char else T5Tokenizer.from_pretrained(pretrained)

    pretrained_model = pretrained
    if args.reload_path:
        assert not args.no_pretrain
        pretrained_model = args.reload_path
    model, config = load_model_old(pretrained_model, args)

    model.config.bos_token_id = BOS_TOKEN_ID

    if args.data_path:
        dataset = datasets.load_from_disk(args.data_path)
    else:
        dataset = load_dataset('dataloading/pretrain_dataset.py', 'de-en')
        # tokenize = partial(tokenize, poisson_lambda=20.0 if char else 3.5, sent_range=((158, 258) if char else (32000, 32100)))
        tok_dataset = dataset.map(tokenize, batched=True, remove_columns=["id", "translation"], num_proc=10)

    print(tok_dataset)
    # def tokenize_function(examples):
    #     return tok(examples['translation'], return_attention_mask=False)

    # tokenized_datasets = datasets.map(
    #     tokenize_function,
    #     batched=True,
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     load_from_cache_file=not data_args.overwrite_cache,
    # )

    training_args = Seq2SeqTrainingArguments("dumped/pretrain/%s/%s/" % (args.model_type, os.environ.get('SLURM_JOB_ID')),
        evaluation_strategy="steps",
        generation_max_length=1024 if char else 512,
        # predict_with_generate=True,
        num_train_epochs=args.epochs,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        adafactor=True,
        learning_rate=0.005,
        weight_decay=0.001,
        warmup_steps=2000,
        disable_tqdm=not args.debug)


    # max_seq_len = 1024 if char else 512
    mask_prob = 0.15
    avg_span_len = 20.0 if char else 3.5
    max_seq_len = 1024 if char else 512
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=max_seq_len,
        noise_density=mask_prob,
        mean_noise_span_length=avg_span_len,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= expanded_inputs_length:
            total_length = (total_length // expanded_inputs_length) * expanded_inputs_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + expanded_inputs_length] for i in range(0, total_length, expanded_inputs_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    dataset = tok_dataset.map(group_texts, batched=True, num_proc=10)


    data_collator = FlaxDataCollatorForT5MLM(
        tokenizer=tok,
        noise_density=mask_prob,
        mean_noise_span_length=avg_span_len,
        input_length=max_seq_len,
        target_length=targets_length,
        pad_token_id=model.config.pad_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
        char=char,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tok,
        args=training_args, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        compute_metrics=compute_metrics
        )

    trainer.evaluation_loop = MethodType(evaluation_loop, trainer)


    if not args.eval_only:
        trainer.train()
    trainer.evaluate()