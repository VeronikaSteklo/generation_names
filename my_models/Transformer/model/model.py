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


def run_epoch(
        model, tokenizer, data_loader,
        optimizer, device=torch.device("cpu"), label_smoothing=0.1,
        train=True
):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0

    progress_bar = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc="Training" if train else "Validation",
        leave=False
    )

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in progress_bar:
            src = batch['src'].to(device).to(device)
            tgt = batch['tgt'].to(device).to(device)

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
                ignore_index=tokenizer.token_2_idx("<pad>"),
                label_smoothing=label_smoothing if train else 0.0
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            progress_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    return total_loss / len(data_loader)


def evaluate_bleu(model, tokenizer, data_loader, device, mode="greedy"):
    model.eval()
    refs = []
    hyps = []

    with torch.no_grad():
        for batch in data_loader:
            src_texts = batch["src_text"]
            tgt_titles = batch["tgt_text"]

            for text, ref_title in zip(src_texts, tgt_titles):
                gen_title = generate_title(model, tokenizer, text, device=device, mode=mode)

                refs.append([ref_title.split()])
                hyps.append(gen_title.split())

    smooth_fn = SmoothingFunction().method1
    bleu = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)
    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    return bleu, bleu1, bleu4


def generate_title(
        model, tokenizer, text, max_len=512, device=torch.device("cpu"),
        mode="greedy", temperature=1.0, top_k=10
):
    """
    Генерация заголовка для текста.
    mode: "greedy", "sampling", "top-k"
    """

    model.eval()
    pad_id = tokenizer.token_2_idx("<pad>")
    bos_id = tokenizer.token_2_idx("<s>")
    eos_id = tokenizer.token_2_idx("</s>")

    src_ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != pad_id).unsqueeze(-2)

    memory = model.encode(src, src_mask)

    ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_pad_mask = (ys != pad_id).unsqueeze(-2)
        tgt_sub_mask = subsequent_mask(ys.size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        out = model.decode(memory, src_mask, ys, tgt_mask)
        logits = model.generator(out[:, -1])

        if mode == "greedy":
            next_word = logits.argmax(dim=-1).item()

        elif mode == "sampling":
            prob = F.softmax(logits / temperature, dim=-1)
            next_word = torch.multinomial(prob, num_samples=1).item()

        elif mode == "top-k":
            topk_prob, topk_idx = torch.topk(logits, top_k, dim=-1)
            topk_prob = F.softmax(topk_prob / temperature, dim=-1)
            next_word = topk_idx[0, torch.multinomial(topk_prob, 1)].item()

        else:
            raise ValueError("mode должен быть 'greedy', 'sampling' или 'top-k'")

        ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)

        if next_word == eos_id:
            break

    gen_text = tokenizer.decode(ys.squeeze().tolist())
    gen_text_for_bleu = " ".join(gen_text.split())

    return gen_text_for_bleu
