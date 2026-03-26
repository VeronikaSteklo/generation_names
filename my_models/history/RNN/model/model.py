import torch
import torch.nn as nn


class TitleRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.RNN(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, title):
        embedded_text = self.dropout(self.embedding(text))
        _, h = self.rnn(embedded_text)

        embedded_title = self.dropout(self.embedding(title))

        output, _ = self.rnn(embedded_title, h)

        return self.fc_out(output)