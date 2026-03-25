import torch
from torch.utils.data import Dataset
from collections import Counter


class TitleVocab:
    def __init__(self, df, min_freq=10):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "<SEP>"}
        self.stoi = {v: k for k, v in self.itos.items()}

        words = []
        for col in ['text', 'title']:
            for text in df[col].dropna():
                words.extend(str(text).lower().split())

        counts = Counter(words)
        for word, freq in counts.items():
            if freq >= min_freq and word not in self.stoi:
                idx = len(self.itos)
                self.stoi[word] = idx
                self.itos[idx] = word

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(w, self.stoi["<UNK>"]) for w in str(text).lower().split()]

    def decode(self, indices):
        return " ".join([self.itos.get(i.item() if torch.is_tensor(i) else i, "<UNK>")
                         for i in indices if i > 4])


class LSTMTitleDataset(Dataset):
    def __init__(self, df, vocab, max_len=100):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []

        for _, row in df.dropna(subset=['text', 'title']).iterrows():
            text_tokens = self.vocab.encode(str(row['text']))
            title_tokens = self.vocab.encode(row['title'])

            full_seq = [vocab.stoi["<SOS>"]] + text_tokens + [vocab.stoi["<SEP>"]] + title_tokens + [
                vocab.stoi["<EOS>"]]

            if len(full_seq) > 5:
                self.samples.append(full_seq)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]

        input_seq = tokens[:-1]
        target_seq = tokens[1:]

        input_seq = input_seq[:self.max_len] + [0] * max(0, self.max_len - len(input_seq))
        target_seq = target_seq[:self.max_len] + [0] * max(0, self.max_len - len(target_seq))

        return torch.tensor(input_seq), torch.tensor(target_seq)
