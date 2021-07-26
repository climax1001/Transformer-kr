import numpy as np
import torch.nn as nn
import torch
from torch import Tensor

from batch import Batch
from constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, TARGET_PAD
from encoder import Encoder
from decoder import Decoder
from model.embed import Embeddings
from model.vocabulary import Vocabulary
from search import greedy


class Model(nn.Module):

    def __init__(self,
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : Embeddings,
                 trg_embed : Embeddings,
                 src_vocab : Vocabulary,
                 trg_vocab : Vocabulary,
                 cfg:dict,
                 in_trg_size:int,
                 out_trg_size:int,
                 ) -> None:
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.src_vocab = src_vocab
        self.trg_embed = trg_embed
        self.trg_vocab = trg_vocab

        self.encoder = encoder
        self.decoder = decoder

        self.bos_index = self.src_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.src_vocab.stoi[EOS_TOKEN]

        self.target_pad = TARGET_PAD

        self.use_cuda = cfg["training"]["use_cuda"]

        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in", True)
        self.just_count_in = model_cfg.get("just_count_in", False)
        self.gaussian_noise = model_cfg.get("gaussian_noise", False)

        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)


        self.future_prediction = model_cfg.get("future_prediction",0)

    def forward(self,
                src : Tensor,
                trg_input : Tensor,
                src_mask : Tensor,
                src_lengths : Tensor,
                trg_mask : Tensor = None,
                src_input : Tensor = None) ->(Tensor, Tensor, Tensor, Tensor):
        encoder_output, encoder_hidden = self.encode(src = src, src_length = src_lengths, src_mask = src_mask)

        unroll_steps = trg_input.size(1)

        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            noise = trg_input.data.new(trg_input.size()).normal_(0,1)

    def encode(self, src: Tensor, src_length : Tensor, src_mask : Tensor) -> (Tensor, Tensor):

        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)
        return encode_output

    def decode(self, encoder_output : Tensor,
               src_mask :Tensor, trg_input :Tensor, trg_mask :Tensor=None) -> (Tensor, Tensor, Tensor, Tensor):

        trg_embed = self.trg_embed(trg_input)
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output, src_mask=src_mask,
                                      trg_mask=trg_mask)
        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function : nn.Module) ->Tensor:

        skel_out , _ = self.forward(
            src=batch.src, trg_input = batch.trg_input,
            src_mask = batch.src_mask, src_lengths=batch.src_lenghts
        )
        batch_loss = loss_function(skel_out, batch.trg)

        if self.gaussian_noise:
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                noise = noise[:,:, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        return batch_loss, noise

    def run_batch(self, batch: Batch, max_output_length : int,) ->(np.array, np.array):

        encoder_output , encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask
        )

        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy) * 1.5)\

        stacked_output, stacked_attention_scores = greedy()