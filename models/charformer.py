from typing import Optional, Union
import warnings
from einops.einops import repeat
import torch
import torch.nn as nn
from einops import rearrange
from transformers import T5Model, T5ForConditionalGeneration, T5Config
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.dummy_pt_objects import StoppingCriteriaList
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList
)

from models.generate import generate_helper
from models.gbst import GBST
import copy

def compute_loss(self, model, inputs, return_outputs=False):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if self.label_smoother is not None and "labels" in inputs:
        labels = inputs.pop("labels")
    else:
        labels = None
    outputs = model(**inputs)

    # print("OOOOOO", outputs)
    print("IIIIII", inputs["decoder_input_ids"][0])
    print("OOOOOO", outputs[0][0].argmax(dim=-1))
    # print("LLLLLL", labels)
    print("LLLLLL", labels[0])
    # Save past state if it exists
    # TODO: this needs to be fixed and made cleaner later.
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    if labels is not None:
        loss = self.label_smoother(outputs, labels)
    else:
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss


class T5WithGBSTEncoder(nn.Module):
    def __init__(self, pretrained_model, freeze, t5_config=None):
        super().__init__()
        if t5_config is not None:
            self.t5 = T5ForConditionalGeneration(t5_config)
        else:
            if pretrained_model is None:
                config = T5Config.from_pretrained('google/byt5-small')
                self.t5 = T5ForConditionalGeneration(config)
            else:
                self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model)

        if pretrained_model == 't5-small':
            self.gbst = GBST(
                num_tokens=384, # 384
                dim=self.t5.model_dim,
                no_conv=True,
                max_block_size=4)
            self.t5.lm_head = nn.Linear(self.t5.model_dim, 384)
        else:
            self.gbst = GBST(
                num_tokens=self.t5.config.vocab_size, # 384
                dim=self.t5.model_dim,
                no_conv=True,
                max_block_size=4)
            self.gbst.token_emb.weight = copy.deepcopy(self.t5.shared.weight)

        if freeze:
            for param in self.t5.parameters():
                param.requires_grad = False
            for param in self.t5.lm_head.parameters():
                param.requires_grad = True

        self.config = self.t5.config


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, labels=None):
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        out = self.t5(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels)

        # print(x.size() for x in out)
        return out

    def generate(self, input_ids, attention_mask=None, **kwargs):
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        decoder_start = torch.full((input_ids.size(0), 1), self.config.bos_token_id).type_as(input_ids)
        out = self.t5.generate(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_input_ids=decoder_start,
            **kwargs)

        # print(out.size())
        return out

    def generate2(self, input_ids, attention_mask=None, **kwargs):
        embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        # decoder_start = torch.full((input_ids.size(0), 1), self.config.bos_token_id).type_as(input_ids)

        encoded = self.t5.encoder(
            inputs_embeds=embed,
            attention_mask=mask,
            return_dict=True
        )

        return generate_helper(self.t5, 
            encoder_hidden_states=encoded.last_hidden_state,
            encoder_attention_mask=mask,
            max_len=kwargs["max_length"],
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            start_token_id=self.config.bos_token_id,
            )
        
        # return self.t5.generate(
        #     inputs_embeds=embed, 
        #     attention_mask=mask,
        #     decoder_input_ids=decoder_start,
        #     **kwargs)




    @classmethod
    def from_pretrained(cls, reload_folder, freeze=False):
        model = cls(None, freeze)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

class Upsampler(nn.Module):
    def __init__(self, vocab_size, dim, ds_factor=4):
        super().__init__()
        self.ds_factor = ds_factor
        self.up = nn.Linear(dim, dim*ds_factor)
        self.pred_layer = nn.Linear(dim, vocab_size)

    def forward(self, batch, orig_len):
        # batch = rearrange(batch, 'b l d -> b d l 1')
        batch = self.up(batch)
        batch = rearrange(batch, 'b l (d m) -> b (l m) d', m=self.ds_factor)
        preds = self.pred_layer(batch[:, :orig_len])
        return preds


class T5WithGBSTFull(nn.Module):
    def __init__(self, pretrained_model, freeze, t5_config=None):
        super().__init__()
        if t5_config is not None:
            self.t5 = T5Model(t5_config)
        else:
            self.t5 = T5Model.from_pretrained(pretrained_model)
        self.dim = self.t5.shared.weight.size(1)

        self.ds_factor = 4
        causal = True
        self.enc_gbst = GBST(
            num_tokens=self.t5.config.vocab_size, 
            dim=self.dim, 
            max_block_size=self.ds_factor, 
            score_consensus_attn=True, 
            causal=False)
            
        
        self.dec_gbst = GBST(
            num_tokens=self.t5.config.vocab_size, 
            dim=self.dim, 
            max_block_size=self.ds_factor, 
            score_consensus_attn=not causal, 
            causal=causal)

        self.config = self.t5.config

        self.upsample = Upsampler(
            vocab_size=self.t5.config.vocab_size, 
            dim=self.dim, 
            ds_factor=self.ds_factor)

        self.decoding_proj_layer = nn.Linear(self.dim, 64*self.ds_factor)
        self.decoding_char_embs = nn.Embedding(self.t5.config.vocab_size, 64)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True)

        self.pred_layer = nn.Linear(128, self.t5.config.vocab_size)


        if freeze:
            for param in self.t5.parameters():
                param.requires_grad = False
            for param in self.t5.lm_head.parameters():
                param.requires_grad = True

    @classmethod
    def from_pretrained(cls, reload_folder, freeze=False, config=None):
        model = cls(None, freeze, config)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        decoder_input_ids=None, 
        decoder_attention_mask=None, 
        labels=None, 
        encoder_hidden_states=None):
        # print(input_ids, attention_mask)
        # orig_len = decoder_input_ids.size(1)
        # embed, mask = self.gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        # dec_embed, dec_mask = self.gbst(decoder_input_ids, mask=decoder_attention_mask.bool() if decoder_attention_mask is not None else None)
        # out = self.t5(
        #     inputs_embeds=embed, 
        #     attention_mask=mask,
        #     decoder_inputs_embeds=dec_embed,
        #     decoder_attention_mask=dec_mask,
        #     return_dict=True)

        # lm_logits = self.upsample(out.last_hidden_state, orig_len)
        enc, enc_mask = self.encode(input_ids, attention_mask)
        lm_logits = self.decode(decoder_input_ids, decoder_attention_mask, enc, enc_mask)

        # loss = None
        # if labels is not None:
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            # loss = None if labels is None else self.t5.compute_loss(labels, lm_logits)


        output = (lm_logits,)
        # return ((loss,) + output) if loss is not None else output
        return output


    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        embed, mask = self.enc_gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        bos_buffer = decoder_input_ids[:, 0].unsqueeze(1).repeat(1, self.ds_factor-1)
        buffered_dec_ids = torch.cat([bos_buffer, decoder_input_ids], dim=1)
        if decoder_attention_mask is not None:
            mask_buffer = torch.ones_like(decoder_attention_mask)[:, :3]
            buffered_dec_mask = torch.cat([mask_buffer, decoder_attention_mask], dim=1)
        
        dec_embed, dec_mask = self.dec_gbst(buffered_dec_ids, mask=buffered_dec_mask.bool() if decoder_attention_mask is not None else None)
        out = self.t5(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_inputs_embeds=dec_embed,
            decoder_attention_mask=dec_mask,
            return_dict=True)

        dec_char_embs = self.decoding_char_embs(decoder_input_ids)
        hidden_proj = self.decoding_proj_layer(out.last_hidden_state)
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, :dec_char_embs.size(1)]], dim=-1)
        lstm_out, _ = self.lstm(concatenated_lstm_input)
        lm_logits = self.pred_layer(lstm_out)


        # print(x.size() for x in out)
        return (lm_logits,)

    def encode(self, input_ids, attention_mask):
        embed, mask = self.enc_gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        out = self.t5.encoder(
            inputs_embeds=embed, 
            attention_mask=mask,
            return_dict=True)
        return out.last_hidden_state, mask

    def decode(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states, encoder_attention_mask):
        assert decoder_input_ids is not None
        assert encoder_hidden_states is not None

        # Add a buffer of bos tokens to the size of the downsampling factor.
        # This is done to ensure that the hidden states do not contain information of 
        # future characters to predict.
        bos_buffer = decoder_input_ids[:, 0].unsqueeze(1).repeat(1, self.ds_factor-1)
        buffered_dec_ids = torch.cat([bos_buffer, decoder_input_ids], dim=1)
        if decoder_attention_mask is not None:
            mask_buffer = torch.ones_like(decoder_attention_mask)[:, :3]
            buffered_dec_mask = torch.cat([mask_buffer, decoder_attention_mask], dim=1)
        
        dec_embed, dec_mask = self.gbst(buffered_dec_ids, mask=buffered_dec_mask.bool() if decoder_attention_mask is not None else None)
        out = self.t5.decoder(
            inputs_embeds=dec_embed, 
            attention_mask=dec_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True)

        # Here we don't need the buffer because the LSTM should predict on the previous character
        # dec_char_embs = self.gbst.token_emb(decoder_input_ids)
        # repeated_out = repeat(out.last_hidden_state, 'b n d -> b (n m) d', m=self.ds_factor)

        dec_char_embs = self.decoding_char_embs(decoder_input_ids)
        hidden_proj = self.decoding_proj_layer(out.last_hidden_state)
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, :dec_char_embs.size(1)]], dim=-1)
        lstm_out, _ = self.lstm(concatenated_lstm_input)
        lm_logits = self.pred_layer(lstm_out)

        # lm_logits = self.upsample(out.last_hidden_state, orig_len)
        return lm_logits


    def decode_lstm_generate(self, decoder_input_ids, decoder_attention_mask, encoder_hidden_states, encoder_attention_mask, lstm_hidden_state):
        dec_embed, dec_mask = self.dec_gbst(decoder_input_ids, mask=decoder_attention_mask.bool() if decoder_attention_mask is not None else None)
        assert encoder_hidden_states is not None
        out = self.t5.decoder(
            inputs_embeds=dec_embed, 
            attention_mask=dec_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True)

        last_hidden_token = out.last_hidden_state[:, -1].unsqueeze(1)

        # dec_char_embs = self.gbst.token_emb(decoder_input_ids[:, -1].unsqueeze(1))
        dec_char_embs = self.decoding_char_embs(decoder_input_ids[:, -1].unsqueeze(1))
        hidden_proj = self.decoding_proj_layer(last_hidden_token)
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        # print(dec_char_embs.size(), out.last_hidden_state.size())
        # concatenated_lstm_input = torch.cat([dec_char_embs, last_hidden_token], dim=-1)
        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, 0].unsqueeze(1)], dim=-1)
        
        all_preds = []
        for i in range(4):
            lstm_out, lstm_hidden_state = self.lstm(concatenated_lstm_input, lstm_hidden_state)
            lm_logits = self.pred_layer(lstm_out)
            preds = lm_logits.argmax(dim=-1)
            all_preds.append(preds)

            if i < 3: # updates for next lstm iteration
                # pred_embs = self.gbst.token_emb(preds)
                pred_embs = self.decoding_char_embs(preds)
                # print(pred_embs.size())
                concatenated_lstm_input = torch.cat([pred_embs, repeated_out[:, i+1].unsqueeze(1)], dim=-1)

        return torch.cat(all_preds, dim=1), lstm_hidden_state

            

    def generate(self, input_ids, attention_mask=None, **kwargs):
        enc, enc_mask = self.encode(input_ids, attention_mask)
        decoder_start = torch.full((input_ids.size(0), 4), self.config.bos_token_id).type_as(input_ids)
                
        return self.greedy_search_lstm(
            input_ids=decoder_start, 
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            **kwargs)
        



    def greedy_search(
        self,
        input_ids: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
    # ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.

            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
            # encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            # encoder_hidden_states = (
            #     model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            # )

        # keep track of which sequences are already finished
        # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        unfinished_sequences = torch.ones(input_ids.shape[0]).to(input_ids.device)
        cur_len = input_ids.shape[-1]

        while True:
            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.decode(
                decoder_input_ids=input_ids,
                decoder_attention_mask=None,
                encoder_hidden_states=model_kwargs['encoder_hidden_states'],
                encoder_attention_mask=model_kwargs['encoder_attention_mask']
            )

            next_tokens_logits = outputs[:, -4:, :]

            # print("Os:", outputs.size())

            # Store scores, attentions and hidden_states when required
            # if return_dict_in_generate:
            #     if output_scores:
            #         scores += (next_token_logits,)
            #     if output_attentions:
            #         decoder_attentions += (
            #             (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
            #         )
            #         if self.config.is_encoder_decoder:
            #             cross_attentions += (outputs.cross_attentions,)

            #     if output_hidden_states:
            #         decoder_hidden_states += (
            #             (outputs.decoder_hidden_states,)
            #             if self.config.is_encoder_decoder
            #             else (outputs.hidden_states,)
            #         )

            # pre-process distribution
            # next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_logits, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                # print(next_tokens.size(), unfinished_sequences.size())
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + (pad_token_id * (1 - unfinished_sequences)).unsqueeze(1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens.long()], dim=-1)
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            cur_len = cur_len + 4

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                # print(next_tokens)
                unfinished = (next_tokens != eos_token_id).min(dim=-1)[0]
                # print(unfinished)
                unfinished_sequences = unfinished_sequences.mul(unfinished.long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return GreedySearchEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return GreedySearchDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        # print(input_ids.size())
        return input_ids
    


    def greedy_search_lstm(
        self,
        input_ids: torch.FloatTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> torch.LongTensor:
    # ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences for models with a language modeling head using greedy decoding.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            stopping_criteria (:obj:`StoppingCriteriaList`, `optional`):
                An instance of :class:`~transformers.StoppingCriteriaList`. List of instances of class derived from
                :class:`~transformers.StoppingCriteria` used to tell if the generation loop should stop.

            max_length (:obj:`int`, `optional`, defaults to 20):
                **DEPRECATED**. Use :obj:`logits_processor` or :obj:`stopping_criteria` directly to cap the number of
                generated tokens. The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            output_attentions (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more details.
            output_hidden_states (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more details.
            output_scores (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return the prediction scores. See ``scores`` under returned tensors for more details.
            return_dict_in_generate (:obj:`bool`, `optional`, defaults to `False`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            synced_gpus (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the :obj:`forward` function of the
                model. If model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput`,
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` or obj:`torch.LongTensor`: A
            :obj:`torch.LongTensor` containing the generated tokens (default behaviour) or a
            :class:`~transformers.generation_utils.GreedySearchDecoderOnlyOutput` if
            ``model.config.is_encoder_decoder=False`` and ``return_dict_in_generate=True`` or a
            :class:`~transformers.generation_utils.GreedySearchEncoderDecoderOutput` if
            ``model.config.is_encoder_decoder=True``.

        Examples::

            >>> from transformers import (
            ... AutoTokenizer,
            ... AutoModelForCausalLM,
            ... LogitsProcessorList,
            ... MinLengthLogitsProcessor,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])

            >>> outputs = model.greedy_search(input_ids, logits_processor=logits_processor)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
        # init values
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # if return_dict_in_generate and self.config.is_encoder_decoder:
            # encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            # encoder_hidden_states = (
            #     model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            # )

        # keep track of which sequences are already finished
        # unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        unfinished_sequences = torch.ones(input_ids.shape[0]).to(input_ids.device)
        cur_len = input_ids.shape[-1]

        lstm_hidden = None
        while True:
            # prepare model inputs
            # model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            with torch.no_grad():
                next_tokens, lstm_hidden = self.decode_lstm_generate(
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=None,
                    encoder_hidden_states=model_kwargs['encoder_hidden_states'],
                    encoder_attention_mask=model_kwargs['encoder_attention_mask'],
                    lstm_hidden_state=lstm_hidden,
                )

            # print(next_tokens.size())
            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                # print(next_tokens.size(), unfinished_sequences.size())
                next_tokens = next_tokens * unfinished_sequences.unsqueeze(1) + (pad_token_id * (1 - unfinished_sequences)).unsqueeze(1)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens.long()], dim=-1)
            # model_kwargs = self._update_model_kwargs_for_generation(
            #     outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            # )
            cur_len = cur_len + 4

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                # print(next_tokens)
                unfinished = (next_tokens != eos_token_id).min(dim=-1)[0]
                # print(unfinished)
                unfinished_sequences = unfinished_sequences.mul(unfinished.long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        # if return_dict_in_generate:
        #     if self.config.is_encoder_decoder:
        #         return GreedySearchEncoderDecoderOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             encoder_attentions=encoder_attentions,
        #             encoder_hidden_states=encoder_hidden_states,
        #             decoder_attentions=decoder_attentions,
        #             cross_attentions=cross_attentions,
        #             decoder_hidden_states=decoder_hidden_states,
        #         )
        #     else:
        #         return GreedySearchDecoderOnlyOutput(
        #             sequences=input_ids,
        #             scores=scores,
        #             attentions=decoder_attentions,
        #             hidden_states=decoder_hidden_states,
        #         )
        # else:
        # print(input_ids.size())
        return input_ids
    
    