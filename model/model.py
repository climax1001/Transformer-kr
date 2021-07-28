import numpy as np
import torch.nn as nn
import torch
from torch import Tensor

from batch import Batch
from constants import BOS_TOKEN, PAD_TOKEN, EOS_TOKEN, TARGET_PAD
from encoder import Encoder, TransformerEncoder
from decoder import Decoder, TransformerDecoder
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

def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0

    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1

    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1 ) * future_prediction + 1

    # Define source embedding
    src_embed = Embeddings(
        **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
        padding_idx=src_padding_idx)

    # Define target linear
    # Linear layer replaces an embedding layer - as this takes in the joints size as opposed to a token
    trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])

    ## Encoder -------
    enc_dropout = cfg["encoder"].get("dropout", 0.) # Dropout
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
           cfg["encoder"]["hidden_size"], \
           "for transformer, emb_size must be hidden_size"

    # Transformer Encoder
    encoder = TransformerEncoder(**cfg["encoder"],
                                 emb_size=src_embed.embedding_dim,
                                 emb_dropout=enc_emb_dropout)

    ## Decoder -------
    dec_dropout = cfg["decoder"].get("dropout", 0.) # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    # Transformer Decoder
    decoder = TransformerDecoder(
        **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
        emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
        trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)

    # Define the model
    model = Model(encoder=encoder,
                  decoder=decoder,
                  src_embed=src_embed,
                  trg_embed=trg_linear,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model