import re
from collections import Counter
import pickle

import my_models.Seq2seq.config as config


def simple_tokenize(text):
    """Разбивает текст на слова и знаки препинания, сохраняя регистр или в lower"""
    return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

class Vocab:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.index2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.n_words = 4

    def build_vocab(self, texts, titles):
        counter = Counter()
        for text in texts:
            counter.update(simple_tokenize(str(text)))
        for title in titles:
            counter.update(simple_tokenize(str(title)))

        for word, count in counter.most_common(config.MAX_VOCAB_SIZE - 4):
            if count >= config.MIN_COUNT:
                if word not in self.word2index:
                    self.word2index[word] = self.n_words
                    self.index2word[self.n_words] = word
                    self.n_words += 1

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)