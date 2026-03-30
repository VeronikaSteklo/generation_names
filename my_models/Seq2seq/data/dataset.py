import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from my_models.Seq2seq.data.vocab import simple_tokenize


class TitleDataset(Dataset):
    def __init__(self, df, vocab, max_src_len=300, max_trg_len=30):
        self.df = df
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.df)

    def text_to_indices(self, text, max_len, add_sos_eos=True):
        tokens = simple_tokenize(str(text))

        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        indices = [self.vocab.word2index.get(t, 1) for t in tokens]

        if add_sos_eos:
            indices = [2] + indices + [3]

        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        src_tensor = self.text_to_indices(row['text'], self.max_src_len, add_sos_eos=False)
        trg_tensor = self.text_to_indices(row['title'], self.max_trg_len, add_sos_eos=True)

        return src_tensor, trg_tensor


def collate_fn(batch):
    """Функция для дополнения батча паддингами"""
    src_batch, trg_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=0, batch_first=True)

    return src_padded, trg_padded