import torch.nn as nn
from torch import Tensor

from model.embed import PositionalEncoding
from model.layers import TransformerDecoderLayer
from utils.helper import freeze_params, subsequent_mask


class Decoder(nn.Module):
    @property
    def output_size(self):
        return self._output_size

class TransformerDecoder(Decoder):
    def __init__(self, num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 trg_size: int = 97,
                 decoder_trg_trg_: bool = True,
                 **kwargs):
        super(TransformerDecoder, self).__init__()
        self._hidden_size = hidden_size
        self._output_size = trg_size
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, decoder_trg_trg=decoder_trg_trg_) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size, mask_count=True)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p = emb_dropout)
        self.output_layer = nn.Linear(hidden_size, trg_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(self,
            trg_embed: Tensor = None,
            encoder_output: Tensor = None,
            src_mask: Tensor = None,
            trg_mask: Tensor = None,
            **kwargs):

        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)
        x = self.emb_dropout(x)

        padding_mask = trg_mask
        sub_mask = subsequent_mask(trg_embed.size(1)).type_as(trg_mask)

        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=sub_mask, padding_mask=padding_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output, x, None, None

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads
        )