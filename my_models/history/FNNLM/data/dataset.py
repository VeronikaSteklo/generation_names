import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
import re
from my_models.history.FNNLM.config import *


class FNNLMDataset(Dataset):
    def __init__(self, csv_file, context_size, vocab=None):
        df = pd.read_csv(csv_file).head(LIMIT)

        def clean_text(text):
            return re.sub(r'[^\w\s]', '', str(text).lower()).split()

        self.data = []
        if vocab is None:
            words_corpus = []
            for _, row in df.iterrows():
                words_corpus.extend(clean_text(row['text']))
                words_corpus.extend(clean_text(row['title']))

            word_counts = Counter(words_corpus)
            filtered_words = sorted([w for w, freq in word_counts.items() if freq >= MIN_WORD_FREQ])

            self.vocab = {word: i + 3 for i, word in enumerate(filtered_words)}
            self.vocab['<PAD>'] = 0
            self.vocab['<UNK>'] = 1
            self.vocab['<END>'] = 2
        else:
            # Если словарь ПЕРЕДАН (для Val), просто используем его
            self.vocab = vocab
        self.rev_vocab = {i: w for w, i in self.vocab.items()}

        for _, row in df.iterrows():
            text_tokens = clean_text(row['text'])[:context_size]
            title_tokens = clean_text(row['title']) + ['<END>']

            if len(text_tokens) < context_size:
                text_tokens = ['<PAD>'] * (context_size - len(text_tokens)) + text_tokens

            current_context = text_tokens
            for target in title_tokens:
                input_ids = [self.vocab.get(w, 1) for w in current_context]
                target_id = self.vocab.get(target, 1)

                self.data.append((torch.tensor(input_ids, dtype=torch.long), target_id))

                current_context = current_context[1:] + [target]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
