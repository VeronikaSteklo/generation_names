import torch
from torch import nn

from my_models.Seq2seq.model.attention import Attention


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(hidden_dim)

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.lstm = nn.LSTM(emb_dim + (hidden_dim * 2), hidden_dim, batch_first=True)

        self.fc_out = nn.Linear(hidden_dim * 3 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs):
        input_tok = input_tok.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_tok))

        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        context = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=2).squeeze(1))

        return prediction, hidden, cell
