from __future__ import unicode_literals, print_function, division

import pickle
from io import open

MAX_VOCAB_SIZE = 30_000


class Vocab:
    """Создаёт словари с частотами слов на основе входных данных"""

    def __init__(self, name):
        self.name = name
        self.word2index = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.word2count = {"<pad>": 0, "<unk>": 0, "<sos>": 0, "<eos>": 0}
        self.index2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.n_words = 4

        self._temp_word_counts = {}

    def addText(self, text: str):
        """Для каждого слова в тексте добавляет его во временный счётчик"""
        for word in text.split():
            self._temp_word_counts[word] = self._temp_word_counts.get(word, 0) + 1

    def build_vocab(self, is_text: bool = False):
        """Строит финальный словарь после подсчёта всех слов"""
        sorted_words = sorted(self._temp_word_counts.items(),
                              key=lambda x: x[1],
                              reverse=True)

        for word, count in sorted_words[:MAX_VOCAB_SIZE - 4]:
            if word not in self.word2index:

                if is_text:
                    if count > 10:
                        self.word2index[word] = self.n_words
                        self.word2count[word] = count
                        self.index2word[self.n_words] = word
                        self.n_words += 1
                    else:
                        self.word2count["<unk>"] += count

                else:
                    if count > 5:
                        self.word2index[word] = self.n_words
                        self.word2count[word] = count
                        self.index2word[self.n_words] = word
                        self.n_words += 1
                    else:
                        self.word2count["<unk>"] += count

        for word, count in sorted_words[MAX_VOCAB_SIZE - 4:]:
            self.word2count["<unk>"] += count

    def word_to_index(self, word: str) -> int:
        """Возвращает индекс слова или <unk>"""
        return self.word2index.get(word, self.word2index["<unk>"])

    def index_to_word(self, index: int) -> str:
        """Возвращает слово по индексу"""
        return self.index2word.get(index, self.word2index["<unk>"])

    def save(self, file_path: str):
        """Сохраняет словарь в файл"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'name': self.name,
                'word2index': self.word2index,
                'word2count': self.word2count,
                'index2word': self.index2word,
                'n_words': self.n_words
            }, f)

    @classmethod
    def load(cls, file_path: str):
        """Загружает словарь из файла"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        vocab = cls(data['name'])
        vocab.word2index = data['word2index']
        vocab.word2count = data['word2count']
        vocab.index2word = data['index2word']
        vocab.n_words = data['n_words']

        return vocab

    def __str__(self):
        """Строковое представление словаря"""
        return (
            f"Vocab(name='{self.name}', "
            f"n_words={self.n_words}, "
        )

    def __len__(self):
        return len(self.word2index)
