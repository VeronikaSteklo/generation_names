import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
import re
from my_models.history.FNNLM.config import *


def first_n_sentences(text, n=2):
    text = str(text)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return ' '.join(sentences[:n])


def get_balanced_sample(df, limit=100000):
    title_counts = df['title'].value_counts()

    max_per_title = max(1, limit // len(title_counts))

    sampled_dfs = []
    for title in title_counts.index:
        title_df = df[df['title'] == title].head(max_per_title)
        sampled_dfs.append(title_df)

    result_df = pd.concat(sampled_dfs).sample(frac=1).reset_index(drop=True)
    return result_df.head(limit)


def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).split()


class FNNLMDataset(Dataset):
    def __init__(self, csv_file, context_size, vocab=None, first_sentences: int = None):
        df = pd.read_csv(csv_file).head(LIMIT)
        df = get_balanced_sample(df, LIMIT)

        self.data = []
        if first_sentences is not None:
            df['text'] = df['text'].apply(lambda x: first_n_sentences(x, first_sentences))

        if vocab is None:
            # Создаём словарь для train
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
            self.vocab = vocab

        self.rev_vocab = {i: w for w, i in self.vocab.items()}

        for _, row in df.iterrows():
            text_tokens = clean_text(row['text'])[:context_size]
            title_tokens = clean_text(row['title']) + ['<END>']

            if len(text_tokens) < context_size:
                text_tokens = ['<PAD>'] * (context_size - len(text_tokens)) + text_tokens

            current_context = text_tokens
            for target in title_tokens:
                input_ids = []
                for w in current_context:
                    if w in self.vocab:
                        input_ids.append(self.vocab[w])
                    else:
                        input_ids.append(self.vocab['<UNK>'])

                if target in self.vocab:
                    target_id = self.vocab[target]
                else:
                    target_id = self.vocab['<UNK>']

                self.data.append((torch.tensor(input_ids, dtype=torch.long), target_id))

                current_context = current_context[1:] + [target if target in self.vocab else '<UNK>']

        if vocab is not None:
            all_words = set()
            for _, row in df.iterrows():
                all_words.update(clean_text(row['text']))
                all_words.update(clean_text(row['title']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
