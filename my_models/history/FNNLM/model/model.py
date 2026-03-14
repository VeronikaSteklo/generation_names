import torch
import torch.nn as nn


class FNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout=0.3):
        super(FNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        self.direct = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embeds = self.embeddings(x)

        pooled = torch.mean(embeds, dim=1)
        pooled = self.dropout(pooled)

        hidden = torch.tanh(self.linear1(pooled))
        hidden = self.dropout(hidden)

        output = self.linear2(hidden) + self.direct(pooled)
        return output