import random
import re
from collections import defaultdict, Counter
from typing import List, Tuple
from nltk.translate.meteor_score import meteor_score
import pickle

import pandas as pd
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    """Функция самой простой предобработки текста, основанной на разбиении на токены по пробелам."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", "", text)
    return text.split()


class TitleNgramModel:
    """Модель n-грамм, обучающаяся по парам (текст → название)."""

    def __init__(self, n_gram: int = 4, smoothing: str | None = 'laplace', alpha: float = 1.):
        self.n_gram = n_gram
        self.ngrams = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocab = set()
        self.start_token = "<s>"
        self.title_token = "<title>"
        self.end_token = "</s>"
        self.unknown_token = "<unk>"
        self.smoothing = smoothing
        self.alpha = alpha
        self.lower_order_models = {}

    def create_ngrams(self, tokens: List[str]) -> list[tuple[str, ...]]:
        """Создание n-грамм с учётом начала и конца."""
        tokens = [self.start_token] + tokens + [self.end_token]
        return [tuple(tokens[i:i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)]

    def train(self, pairs: List[Tuple[str, str]]) -> None:
        """Обучение на списке пар (text, title)"""
        all_tokens = []
        for text, title in pairs:
            text_tokens = tokenize(text)
            title_tokens = tokenize(title)
            combined = text_tokens + [self.title_token] + title_tokens
            all_tokens.extend(combined)

        token_counts = Counter(all_tokens)
        self.vocab = set(token for token, count in token_counts.items() if count > 1)
        self.vocab.update([self.start_token, self.title_token, self.end_token, self.unknown_token])

        if self.n_gram > 1:
            for n in range(1, self.n_gram):
                self.lower_order_models[n] = TitleNgramModel(n, self.smoothing, self.alpha)
                self.lower_order_models[n].train(pairs)

        for text, title in pairs:
            text_tokens = tokenize(text)
            title_tokens = tokenize(title)

            text_tokens = [token if token in self.vocab else self.unknown_token for token in text_tokens]
            title_tokens = [token if token in self.vocab else self.unknown_token for token in title_tokens]

            combined = text_tokens + [self.title_token] + title_tokens
            ngrams_list = self.create_ngrams(combined)

            for ngram in ngrams_list:
                context = ngram[:-1]
                next_word = ngram[-1]
                self.ngrams[context][next_word] += 1
                self.context_counts[context] += 1

    def train_from_csv(self, csv_path: str, text_col: str = "text", title_col: str = "title",
                       limit: int | None = None):
        """Обучение модели по CSV-файлу"""
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)

        pairs = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обучение модели"):
            text = getattr(row, text_col, None)
            title = getattr(row, title_col, None)
            if text and title:
                pairs.append((text, title))

        self.train(pairs)

        print(f"Обучение завершено. Количество уникальных контекстов: {len(self.ngrams)}")

    def get_simple_probability(self, context: tuple, word: str) -> float:
        """Вычисляет вероятность по формуле P(w_t | context) = C(context + word) / C(context)"""
        context_counts = self.ngrams.get(context, {})
        total = sum(context_counts.values())
        return context_counts[word] / total if total else 0.0

    def get_laplace_probability(self, context: tuple, word: str) -> float:
        """Сглаживание Лапласа (Add-one)."""
        context_counts = self.ngrams.get(context, {})
        total = sum(context_counts.values())
        vocab_size = len(self.vocab)

        count_word = context_counts.get(word, 0)
        return (count_word + self.alpha) / (total + self.alpha * vocab_size)

    def get_linear_interpolation_probability(self, context: tuple, word: str) -> float:
        """Линейная интерполяция с моделями меньшего порядка."""
        if self.n_gram == 1 or not self.lower_order_models:
            return self.get_laplace_probability(context, word)

        lambda_current = 0.6
        lambda_backoff = 0.4

        current_prob = self.get_laplace_probability(context, word)

        backoff_context = context[1:] if len(context) > 1 else tuple()
        backoff_model = self.lower_order_models[self.n_gram - 1]
        backoff_prob = backoff_model.get_probability(backoff_context, word)

        return lambda_current * current_prob + lambda_backoff * backoff_prob

    def get_probability(self, context: tuple, word: str) -> float:
        """Вычисляет вероятность с выбранным методом сглаживания."""

        if word not in self.vocab:
            word = self.unknown_token

        if self.smoothing == 'laplace':
            return self.get_laplace_probability(context, word)
        elif self.smoothing == 'linear':
            return self.get_linear_interpolation_probability(context, word)
        elif self.smoothing is None:
            return self.get_simple_probability(context, word)
        else:
            return self.get_laplace_probability(context, word)

    def generate_with_backoff(self, context: tuple, max_depth: None | int = None):
        """Генерация с backoff: если контекст не найден, используем контекст короче."""

        if max_depth is None:
            max_depth = self.n_gram - 1

        current_context = context
        depth = 0

        while depth <= max_depth:
            if current_context in self.ngrams:
                probabilities = dict()

                for word in self.vocab:
                    if word not in [self.title_token, self.title_token, self.end_token]:
                        probabilities[word] = self.get_probability(current_context, word)

                total_prob = sum(probabilities.values())
                if total_prob > 0:
                    return {word: prob / total_prob for word, prob in probabilities.items()}

            if len(current_context) > 1:
                current_context = current_context[1:]
            else:
                break
            depth += 1

        words = [w for w in self.vocab if w not in [self.start_token, self.title_token, self.end_token]]
        return {word: 1.0 / len(words) for word in words}

    def generate_title(self, input_text: str, max_words: int = 7) -> str:
        """Генерация названия на основе входного текста с использованием backoff и сглаживания."""
        tokens = tokenize(input_text)
        tokens = [token if token in self.vocab else self.unknown_token for token in tokens]
        tokens += [self.title_token]
        context = tuple(tokens[-(self.n_gram - 1):])
        result = []

        for _ in range(max_words):
            word_probs = self.generate_with_backoff(context)

            if not word_probs:
                break

            word_probs = {word: prob for word, prob in word_probs.items()}
            total = sum(word_probs.values())
            word_probs = {word: prob / total for word, prob in word_probs.items()}
            words = list(word_probs.keys())
            probs = list(word_probs.values())

            next_word = random.choices(words, weights=probs)[0]

            if next_word in {self.end_token, self.title_token}:
                break

            result.append(next_word)
            context = (*context[1:], next_word) if len(context) > 0 else (next_word,)

        title = " ".join(result)
        return title if title else "Без названия"

    def evaluate_meteor(self, test_pairs: List[Tuple[str, str]]) -> float:
        """Вычисление средней METEOR-оценки для тестового набора (text, reference_title)."""
        scores = []
        for text, true_title in tqdm(test_pairs, desc="Оценка METEOR"):
            generated = self.generate_title(text)
            score = meteor_score([tokenize(true_title)], tokenize(generated))
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

train_df = "../../data/training_data/train_df.csv"
val_df = pd.read_csv("../../data/training_data/val_df.csv")

model = TitleNgramModel(n_gram=3)
model.train_from_csv(train_df)
model.save(path="../../models/history/title_ngram_model.pkl")

val_pairs = list(zip(val_df.text, val_df.title))
meteor = model.evaluate_meteor(val_pairs)
print(f"\nСредняя METEOR-оценка на валидационном наборе: {meteor:.4f}")
