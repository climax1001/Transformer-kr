import torch.nn as nn
import torch

from encoder import Encoder
from decoder import Decoder
class Model(nn.Module):

    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed):