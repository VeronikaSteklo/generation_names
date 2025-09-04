from typing import List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def text_to_tensor(text: str, vocab, add_sos_eos=True, max_len: int | None = None,
                   truncate_from_start=False) -> torch.Tensor:
    """Преобразует текст в тензоры, с опциональной обрезкой"""
    tokens = text.strip().split()

    if max_len is not None:
        if truncate_from_start:
            tokens = tokens[-max_len:]
        else:
            tokens = tokens[:max_len]

    indices = [vocab.word_to_index(w) for w in tokens]

    if add_sos_eos:
        indices = [vocab.word2index["<sos>"]] + indices + [vocab.word2index["<eos>"]]

    return torch.tensor(indices, dtype=torch.long)


class TitleDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], input_vocab, output_vocab):
        """
            pairs — список пар (название, текст),
            input_vocab - словарь с частотами слов из текстов,
            output_vocab - словарь с частотами слов из названий
        """
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        title, text = self.pairs[idx]
        input_tensor = text_to_tensor(text, self.input_vocab, add_sos_eos=False, max_len=300)
        target_tensor = text_to_tensor(title, self.output_vocab, add_sos_eos=False, max_len=30)
        return input_tensor, target_tensor


def collate_fn(batch: List[tuple[str, str]]):
    """
    batch: list of (input_tensor, target_tensor)
    Returns:
        input_padded: [batch, src_len]
        target_padded: [batch, trg_len]
    """
    src_batch, trg_batch = zip(*batch)

    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=0, batch_first=True)

    return src_padded, trg_padded
