import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        input = input.unsqueeze(0)  # [1, B]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell