# generate.py
import torch

def generate_helper(model, 
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        return_dict=None,
        max_len=200,
        pad_token_id=None,
        eos_token_id=None,
        start_token_id=None):
        """
        Decode a sentence given initial start.
        `x`:
            - LongTensor(bs, slen)
                <EOS> W1 W2 W3 <EOS> <PAD>
                <EOS> W1 W2 W3   W4  <EOS>
        `lengths`:
            - LongTensor(bs) [5, 6]
        `positions`:
            - False, for regular "arange" positions (LM)
            - True, to reset positions from the new generation (MT)
        """
        src_enc = encoder_hidden_states

        pad_token_id = model.pad_index if None else pad_token_id
        start_token_id = model.bos_index if None else start_token_id
        eos_token_id = model.eos_index if None else eos_token_id

        # input batch
        bs = src_enc.size(0)

        # generated sentences
        generated = torch.zeros((bs, max_len), dtype=torch.long, device=src_enc.device)
        # positions
        # positions = torch.arange(max_len, dtype=torch.long, device=src_enc.device).unsqueeze(0).expand(bs, max_len)
        # current position / max lengths / length of generated sentences / unfinished sentences
        cur_len = 1
        unfinished_sents = torch.full((bs,), 1, dtype=torch.long, device=src_enc.device)

        # cache compute states
        while cur_len < max_len:
            # compute word scores
            out = model.decoder.forward(
                input_ids=generated[:, :cur_len],
                encoder_hidden_states=src_enc,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True
            )
            logits = model.lm_head.forward(out.last_hidden_state)

            # out.logits: bs, slen, n_words
            scores = logits[:, -1, :]

            # select next words: sample or greedy
            next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)

            # update generations / lengths / finished sentences / current length
            generated[:, cur_len] = next_words * unfinished_sents + pad_token_id * (1 - unfinished_sents)

            unfinished_sents *= (next_words.ne(eos_token_id).long())
            cur_len = cur_len + 1

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

        # add <EOS> to unfinished sentences
        if cur_len == max_len:
            generated[:, -1].masked_fill_(unfinished_sents.bool(), eos_token_id)

        # sanity check
        assert (generated == eos_token_id).sum() == bs

        return generated[:, :cur_len]#, gen_len    





