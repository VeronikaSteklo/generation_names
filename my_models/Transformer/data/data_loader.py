import re

import pandas as pd
import torch
from torch.utils.data import Dataset


def clean_text(text: str) -> str:
    """Очистка текста: нижний регистр + удаление пунктуации кроме тире (-) и двоеточия (:)"""
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9:\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextTitleDataset(Dataset):
    def __init__(
            self, data_path, text_fill="text",
            title_fill="title", tokenizer=None, max_text_len=512,
            max_title_len=25, use_clean=True, limit=None
    ):
        self.data = pd.read_csv(data_path)
        if limit:
            self.data = self.data.head(limit)
        print(f"Количество данных в исходном датасете: {len(self.data)}")
        self.text_fill = text_fill
        self.title_fill = title_fill
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_title_len = max_title_len
        self.use_clean = use_clean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        text = str(row[self.text_fill])
        title = str(row[self.title_fill])

        if self.use_clean:
            text = clean_text(text)
            title = clean_text(title)

        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        title_tokens = self.tokenizer.encode(title, add_special_tokens=True)

        text_tokens = text_tokens[:self.max_text_len]
        title_tokens = title_tokens[:self.max_title_len]

        return {
            "src": torch.tensor(text_tokens, dtype=torch.long),
            "tgt": torch.tensor(title_tokens, dtype=torch.long),
            "src_text": text,
            "tgt_text": title
        }


def collate_fn(batch, pad_id, device=torch.device("cpu")):
    """Собирает батч и паддит последовательности"""
    text_seq = [item["src"] for item in batch]
    title_seq = [item["tgt"] for item in batch]
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]

    max_text_len = max(len(text) for text in text_seq)
    max_title_len = max(len(title) for title in title_seq)

    text_batch = torch.full((len(batch), max_text_len), pad_id, dtype=torch.long)
    title_batch = torch.full((len(batch), max_title_len), pad_id, dtype=torch.long)

    for i, seq in enumerate(text_seq):
        text_batch[i, :len(seq)] = seq
    for i, seq in enumerate(title_seq):
        title_batch[i, :len(seq)] = seq

    return {
        "src": text_batch.to(device),
        "tgt": title_batch.to(device),
        "src_text": src_texts,
        "tgt_text": tgt_texts
    }
