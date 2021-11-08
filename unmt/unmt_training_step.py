# unmt_training_step.py
import torch

def get_batch_lens_by_eos(batch, eos_token_id):
    batch_idx, eos_idx = (batch == eos_token_id).nonzero(as_tuple=True)
    last_idx = -1
    lens = torch.full((batch.size(0),), batch.size(1), dtype=torch.long, device=batch.device)
    for i, idx in enumerate(batch_idx):
        if idx != last_idx:
            lens[idx] = eos_idx[i]      
            last_idx = idx

    return lens

def get_attn_mask(batch, batch_lens):
    return torch.arange(batch.size(1), dtype=torch.long, device=batch.device) < batch_lens[:, None]


def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None

    gen_kwargs = {
        "max_length": self._max_length if self._max_length is not None else self.model.config.max_length
    }

    with torch.no_grad():
        self.model.eval()
        gen_tokens_tgt = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            decoder_start_token_id=self.LANG2_TOKEN_ID,
            max_length=int(1.3 * inputs["input_ids"].size(1) + 5) 
        )
        gen_tokens_src = self.model.generate(
            inputs["decoder_input_ids"],
            attention_mask=inputs["decoder_attention_mask"],
            decoder_start_token_id=self.LANG1_TOKEN_ID,
            max_length=int(1.3 * inputs["decoder_input_ids"].size(1) + 5) 
        )
        self.model.train()

    gen_lens_tgt = get_batch_lens_by_eos(gen_tokens_tgt, self.tokenizer.eos_token_id)
    gen_mask_tgt = get_attn_mask(gen_tokens_tgt, gen_lens_tgt)
    gen_lens_src = get_batch_lens_by_eos(gen_tokens_src, self.tokenizer.eos_token_id)
    gen_mask_src = get_attn_mask(gen_tokens_src, gen_lens_src)


    src_labels = inputs["input_ids"][:, 1:].contiguous()
    src_labels[src_labels == 0] = -100

    tgt_labels = inputs["decoder_input_ids"][:, 1:].contiguous()
    tgt_labels[tgt_labels == 0] = -100

    outputs = model(
        input_ids=gen_tokens_tgt,
        attention_mask=gen_mask_tgt,
        decoder_input_ids=inputs["input_ids"][:, :-1],
        decoder_attention_mask=inputs["attention_mask"][:, :-1],
        labels=src_labels
    )

    outputs2 = model(
        input_ids=gen_tokens_src,
        attention_mask=gen_mask_src,
        decoder_input_ids=inputs["decoder_input_ids"][:, :-1],
        decoder_attention_mask=inputs["decoder_attention_mask"][:, :-1],
        labels=tgt_labels
    )

    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        loss = self.label_smoother(outputs, labels)
    else:
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] + outputs2["loss"] if isinstance(outputs, dict) else outputs[0] + outputs2[0]

    return (loss, outputs) if return_outputs else loss