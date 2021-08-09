import copy

import torch
import torch.nn as nn
from model.embed import PositionalEncoding
from model.layers import TransformerEncoderLayer
from model.sublayer import Norm
from utils.helper import freeze_params
from torchsummary import summary

def get_clones(module , N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    @property
    def output_size(self):
        return self._output_size

class TransformerEncoder(Encoder):
    def __init__(self, hidden_size : int = 512, ff_size : int = 2048
                 ,num_layers: int = 8, num_heads: int = 4, dropout: float = 0.1,
                 emb_dropout: float = 0.1, freeze: bool = False, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    def forward(self, embed_src, src_length, mask):
        x = embed_src
        x = self.pe(x)
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x), None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)




if __name__ == '__main__':
    print(TransformerEncoder())