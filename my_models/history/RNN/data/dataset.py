import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import re
from collections import Counter
import my_models.history.RNN.config as config


class Vocab:
    def __init__(self, texts, min_freq=5):
        self.itos = [config.PAD_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN, config.UNK_TOKEN]
        self.stoi = {token: i for i, token in enumerate(self.itos)}

        tokens = []
        for text in texts:
            tokens.extend(self.tokenize(str(text)))

        counter = Counter(tokens)
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    @staticmethod
    def tokenize(text):
        return re.findall(r'\w+', text.lower())

    def encode(self, text):
        return [self.stoi.get(token, self.stoi[config.UNK_TOKEN]) for token in self.tokenize(str(text))]
    
    def __len__(self):
        return len(self.itos)


class TitleDataset(Dataset):
    def __init__(self, df, vocab):
        self.text = df['text'].astype(str).values
        self.titles = df['title'].astype(str).values
        self.vocab = vocab

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        title = self.titles[idx]

        lec_tensor = torch.tensor(self.vocab.encode(text), dtype=torch.long)

        title_encoded = self.vocab.encode(title)
        title_tensor = torch.tensor(
            [self.vocab.stoi[config.SOS_TOKEN]] +
            title_encoded +
            [self.vocab.stoi[config.EOS_TOKEN]],
            dtype=torch.long
        )

        return lec_tensor, title_tensor


def collate_fn(batch, pad_idx):
    lecs, titles = zip(*batch)
    lecs_padded = pad_sequence(lecs, batch_first=True, padding_value=pad_idx)
    titles_padded = pad_sequence(titles, batch_first=True, padding_value=pad_idx)
    return lecs_padded, titles_padded
