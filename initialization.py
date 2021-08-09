# coding: utf-8



import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out


def xavier_uniform_n_(w: Tensor, gain: float = 1., n: int = 4) -> None:

    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


# pylint: disable=too-many-branches
def initialize_model(model: nn.Module, cfg: dict, src_padding_idx: int,
                     trg_padding_idx: int) -> None:

    # defaults: xavier, embeddings: normal 0.01, biases: zeros, no orthogonal
    gain = float(cfg.get("init_gain", 1.0))  # for xavier
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))

    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # for xavier

    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    # pylint: disable=unnecessary-lambda, no-else-return
    def _parse_init(s, scale, _gain):
        scale = float(scale)
        assert scale > 0., "incorrect init_weight"
        if s.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        elif s.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif s.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0., std=scale)
        elif s.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():

            if "embed" in name:
                if "bias" in name:
                    bias_init_fn_(p)
                else:
                    embed_init_fn_(p)

            elif "bias" in name:
                bias_init_fn_(p)

            elif len(p.size()) > 1:

                # RNNs combine multiple matrices is one, which messes up
                # xavier initialization
                if init == "xavier" and "rnn" in name:
                    n = 1
                    if "encoder" in name:
                        n = 4 if isinstance(model.encoder.rnn, nn.LSTM) else 3
                    elif "decoder" in name:
                        n = 4 if isinstance(model.decoder.rnn, nn.LSTM) else 3
                    xavier_uniform_n_(p.data, gain=gain, n=n)
                else:
                    init_fn_(p)

        # zero out paddings
        model.src_embed.lut.weight.data[src_padding_idx].zero_()