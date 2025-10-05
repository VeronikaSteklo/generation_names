import random
import re
from collections import defaultdict, Counter
from typing import List


def tokenize(text: str) -> List[str]:
    """Функция самой простой предобработки текста, основанной на разбиении на токены по пробелам."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", "", text)
    text = text.split()
    return text


class NgramModel:
    """Базовая модель для генерации текста на основе n-грамм"""

    def __init__(self, n_gram: int = 2):
        self.n_gram = n_gram
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.start_token = "<s>"
        self.end_token = "</s>"

    def create_ngrams(self, text: str | List[str]) -> List[tuple]:
        """Создаёт список n-грамм (включая start и end токены)."""
        if type(text) == str:
            text = tokenize(text)
        tokens = [self.start_token] + text + [self.end_token]
        ngrams_list = []
        for i in range(len(tokens) - self.n_gram + 1):
            ngram = tuple(tokens[i:i + self.n_gram])
            ngrams_list.append(ngram)
        return ngrams_list

    def train(self, text: str) -> None:
        """Обучает модель по входному тексту."""
        tokens = tokenize(text)
        ngrams_list = self.create_ngrams(tokens)
        self.vocab.update(tokens)

        for ngram in ngrams_list:
            context = ngram[:-1]
            next_word = ngram[-1]
            self.ngrams[context][next_word] += 1

    def get_probability(self, context: tuple, word: str) -> float:
        """Вычисляет вероятность по формуле P(w_t | context) = C(context + word) / C(context)"""
        context_counts = self.ngrams.get(context, {})
        total_context = sum(context_counts.values())
        if total_context == 0:
            return 0.0
        return context_counts[word] / total_context

    def generate(self, max_words: int = 10) -> str:
        """Генерация текста по вероятностной формуле"""
        possible_starts = [ctx for ctx in self.ngrams.keys() if ctx[0] == self.start_token]
        if not possible_starts:
            return ""
        context = random.choice(possible_starts)
        result = [w for w in context if w != self.start_token]

        for _ in range(max_words):
            next_candidates = self.ngrams.get(context)
            if not next_candidates:
                break

            words = list(next_candidates.keys())
            probs = [self.get_probability(context, w) for w in words]

            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                break

            next_word = random.choices(words, weights=probs)[0]

            if next_word == self.end_token:
                break

            result.append(next_word)
            context = (*context[1:], next_word)

        return " ".join(result).capitalize()
