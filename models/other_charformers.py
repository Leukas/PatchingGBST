# other_charformers.py
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from einops import rearrange
from models.causal_gbst_fast import GBST as MaskedGBST
from models.charformer_bert import BertWithGBSTEncDec

class BertWithMaskedGBSTEncDec(BertWithGBSTEncDec):
    def __init__(self, pretrained_model, freeze, bert_config=None, ds_factor=4):
        super().__init__(pretrained_model, freeze, bert_config, False, ds_factor)

        self.enc_gbst = MaskedGBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=4, 
            score_consensus_attn=True
            )
            
        self.dec_gbst = MaskedGBST(
            num_tokens=self.vocab_size, 
            dim=self.dim, 
            downsample_factor=self.ds_factor,
            max_block_size=self.ds_factor, 
            score_consensus_attn=True 
            )

    @classmethod
    def from_pretrained(cls, reload_folder, freeze=False, config=None, ds_factor=4):
        model = cls(None, freeze, config, ds_factor)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model


class BertWithPaddedGBSTEncDec(BertWithGBSTEncDec):
    def __init__(self, pretrained_model, freeze, bert_config=None, ds_factor=4):
        super().__init__(pretrained_model, freeze, bert_config, False, ds_factor)

    @classmethod
    def from_pretrained(cls, reload_folder, freeze=False, config=None, ds_factor=4):
        model = cls(None, freeze, config, ds_factor)
        state_dict_path = reload_folder + "/pytorch_model.bin"
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        embed, mask = self.enc_gbst(input_ids, mask=attention_mask.bool() if attention_mask is not None else None)
        bos_buffer = decoder_input_ids[:, 0].unsqueeze(1).repeat(1, 2*self.ds_factor-1)
        buffered_dec_ids = torch.cat([bos_buffer, decoder_input_ids], dim=1)
        if decoder_attention_mask is not None:
            mask_buffer = torch.ones_like(decoder_attention_mask)[:, :3]
            buffered_dec_mask = torch.cat([mask_buffer, decoder_attention_mask], dim=1)
        
        dec_embed, dec_mask = self.dec_gbst(buffered_dec_ids, mask=buffered_dec_mask.bool() if decoder_attention_mask is not None else None)
        out = self.bert(
            inputs_embeds=embed, 
            attention_mask=mask,
            decoder_inputs_embeds=dec_embed,
            decoder_attention_mask=dec_mask,
            output_hidden_states=True,
            return_dict=True)

        dec_char_embs = self.decoding_char_embs(decoder_input_ids)
        hidden_proj = self.decoding_proj_layer(out.decoder_hidden_states[-1])
        repeated_out = rearrange(hidden_proj, 'b n (m d) -> b (n m) d', m=self.ds_factor)

        concatenated_lstm_input = torch.cat([dec_char_embs, repeated_out[:, :dec_char_embs.size(1)]], dim=-1)
        lstm_out, _ = self.lstm(concatenated_lstm_input)
        lm_logits = self.pred_layer(lstm_out)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=out.past_key_values,
                decoder_hidden_states=out.decoder_hidden_states,
                decoder_attentions=out.decoder_attentions,
                cross_attentions=out.cross_attentions,
                encoder_last_hidden_state=out.encoder_last_hidden_state,
                encoder_hidden_states=out.encoder_hidden_states,
                encoder_attentions=out.encoder_attentions,
                )

        return (lm_logits,)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        enc, enc_mask = self.encode(input_ids, attention_mask)
        
        bos_token_id = \
            kwargs["decoder_start_token_id"] if "decoder_start_token_id" in kwargs \
            else self.config.bos_token_id # NOTE: check this line if it works with nmt.py, otherwise add decoder_start_token_id to kwargs
        
        assert self.config.bos_token_id == bos_token_id, str(self.config.bos_token_id) + ", " + str(bos_token_id) # To remove from non-nmt.py runs

        decoder_start = torch.full((input_ids.size(0), 2*self.ds_factor), bos_token_id).type_as(input_ids)
                
        return self.greedy_search_lstm(
            input_ids=decoder_start, 
            encoder_hidden_states=enc,
            encoder_attention_mask=enc_mask,
            **kwargs)
        