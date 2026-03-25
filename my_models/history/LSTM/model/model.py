import torch.nn as nn


class LSTMTitleGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, hidden=None):
        x = self.embedding(x)

        out, hidden = self.lstm(x, hidden)

        out = self.ln(out)
        out = self.dropout(out)

        logits = self.fc(out)
        return logits, hidden
