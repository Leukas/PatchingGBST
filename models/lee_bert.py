# lee_bert.py
import torch
import torch.nn as nn
from models.lee import CharToPseudoWord
from models.charformer_bert import BertWithGBSTEncDec


class CharEmbedder(nn.Module):
    def __init__(self, vocab_size, char_emb_dim, dim, ds_factor, decoder=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, char_emb_dim)
        self.dropout = nn.Dropout(0.1)
        self.pos = nn.Parameter(    
            torch.randn(1, 1024, char_emb_dim))
        self.lee = CharToPseudoWord(
            char_emb_dim, 
            intermediate_dim=dim,
            max_pool_window=ds_factor,
            is_decoder=decoder)

    def forward(self, x, mask):
        if mask is None:
            mask = torch.ones((x.size(0), x.size(1)), device=x.device)
        x = self.dropout(self.emb(x)) + self.pos[:, :x.size(1)]
        return self.lee(x, mask.float())

class BertWithLeeFull(BertWithGBSTEncDec):
    def __init__(self, pretrained_model, freeze, bert_config=None, causal=True, ds_factor=4):
        super().__init__(pretrained_model, freeze, bert_config, causal, ds_factor)

        self.enc_gbst = CharEmbedder(self.vocab_size, 64, self.dim, self.ds_factor, False)
        self.dec_gbst = CharEmbedder(self.vocab_size, 64, self.dim, self.ds_factor, True)

    
    