import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.bool
    )
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model,
                 vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x.to(self.lut.weight.device)) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def generate_title(
        model, tokenizer, text, max_len=128, device=torch.device("cpu"),
        mode="beam", temperature=1.0, top_k=10, beam_size=3
):
    """
    Генерация заголовка для текста.
    mode: "greedy", "sampling", "top-k", "beam"
    """

    model.eval()
    pad_id = tokenizer.token_2_idx("<pad>")
    bos_id = tokenizer.token_2_idx("<s>")
    eos_id = tokenizer.token_2_idx("</s>")

    src_ids = tokenizer.encode(text, add_special_tokens=False)[:max_len]
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_mask = (src != pad_id).unsqueeze(-2)

    memory = model.encode(src, src_mask)

    if mode == "beam":
        beams = [(torch.tensor([[bos_id]], device=device), 0.0)]

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                tgt_pad_mask = (seq != pad_id).unsqueeze(-2)
                tgt_sub_mask = subsequent_mask(seq.size(1)).to(device)
                tgt_mask = tgt_pad_mask & tgt_sub_mask

                out = model.decode(memory, src_mask, seq, tgt_mask)
                logits = model.generator(out[:, -1])
                log_probs = F.log_softmax(logits / temperature, dim=-1)

                top_log_probs, top_tokens = torch.topk(log_probs, beam_size, dim=-1)

                for i in range(beam_size):
                    next_seq = torch.cat([seq, top_tokens[:, i].unsqueeze(0)], dim=1)
                    next_score = score + top_log_probs[0, i].item()
                    new_beams.append((next_seq, next_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

            if all(eos_id in seq for seq, _ in beams):
                break

        ys = beams[0][0]

    else:
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
                raise ValueError("mode должен быть 'greedy', 'sampling', 'top-k', 'beam'")

            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)

            if next_word == eos_id:
                break

    gen_text = tokenizer.decode(ys.squeeze().tolist())
    gen_text_for_bleu = " ".join(gen_text.split())

    return gen_text_for_bleu
