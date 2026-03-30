import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden_repeated = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attention_weights = self.v(energy).squeeze(2)

        return F.softmax(attention_weights, dim=1)
