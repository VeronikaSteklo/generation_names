import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)

        return outputs, hidden, cell
