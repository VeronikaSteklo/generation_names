import torch
import torch.nn as nn


class FNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(FNNLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        self.direct = nn.Linear(context_size * embedding_dim, vocab_size)

    def forward(self, x):
        # Добавь это временно в метод forward, чтобы увидеть, какое число ломает код
        if x.max() >= self.embeddings.num_embeddings:
            print(f"Ошибка! Максимальный индекс в батче: {x.max()}")
            print(f"Размер словаря в модели: {self.embeddings.num_embeddings}")
        embeds = self.embeddings(x).view(x.shape[0], -1)

        hidden = torch.tanh(self.linear1(embeds))

        output = self.linear2(hidden) + self.direct(embeds)
        return output
