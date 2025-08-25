import copy

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.functional import log_softmax

import torch.nn.functional as F

from my_models.Transformer.model.attention import MultiHeadedAttention
from my_models.Transformer.model.decoder import Decoder, DecoderLayer
from my_models.Transformer.model.encoder import Encoder, EncoderLayer
from my_models.Transformer.model.utils import PositionwiseFeedForward, PositionalEncoding, Embeddings, subsequent_mask
from my_models.Transformer.tokenization import TikTokenizer


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


def make_model(
        N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, device="cpu"
):
    "Helper: Construct a model from hyperparameters."
    tokenizer = TikTokenizer()
    src_embed = nn.Sequential(
        Embeddings(d_model, tokenizer.vocab_size),
        PositionalEncoding(d_model, dropout)
    )

    tgt_embed = nn.Sequential(
        Embeddings(d_model, tokenizer.vocab_size),
        PositionalEncoding(d_model, dropout)
    )
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed,
        tgt_embed,
        Generator(d_model, tokenizer.vocab_size),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model.to(device), tokenizer


def train_epoch(model, tokenizer, data_loader, optimizer, device=torch.device("cpu")):
    model.train()
    total_loss = 0

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training", leave=False)

    for batch_idx, batch in progress_bar:
        src = batch['src'].to(device).to(device)
        tgt = batch['tgt'].to(device).to(device)

        optimizer.zero_grad()
        tgt_input = tgt[:, :-1]

        tgt_mask = subsequent_mask(tgt_input.size(1)).to(device)
        src_mask = (src != tokenizer.token_2_idx("<pad>")).unsqueeze(-2).to(device)

        out = model(src, tgt_input, src_mask, tgt_mask)
        logits = model.generator(out)
        logits = logits.to(device).contiguous().view(-1, logits.size(-1)).float()

        tgt_labels = tgt[:, 1:].to(device)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_labels.reshape(-1),
            ignore_index=tokenizer.token_2_idx("<pad>")
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    return total_loss / len(data_loader)

