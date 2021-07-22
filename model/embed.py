import torch
import torch.nn as nn
import math

from torch.autograd import Variable

class PositionalEncoding(nn.Module):

    def __init__(self,
                 size: int = 0,
                 max_len: int = 200000, # Max length was too small for the required length
                 mask_count=False):

        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros(max_len, size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, size, 2, dtype=torch.float) *
                              -(math.log(10000.0) / size)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size
        self.mask_count = mask_count

    def forward(self, emb):

        return emb + self.pe[:, :emb.size(1)]
