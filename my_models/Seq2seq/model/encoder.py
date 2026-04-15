import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))

        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))

        return outputs, hidden.unsqueeze(0), cell.unsqueeze(0)
