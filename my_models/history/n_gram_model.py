import random
import re
from collections import defaultdict, Counter
from typing import List, Tuple
from nltk.translate.meteor_score import meteor_score

import pandas as pd
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    """Функция самой простой предобработки текста, основанной на разбиении на токены по пробелам."""
    text = text.lower()
    text = re.sub(r"[^a-zа-яё0-9\s]", "", text)
    return text.split()


class TitleNgramModel:
    """Модель n-грамм, обучающаяся по парам (текст → название)."""

    def __init__(self, n_gram: int = 3):
        self.n_gram = n_gram
        self.ngrams = defaultdict(Counter)
        self.vocab = set()
        self.start_token = "<s>"
        self.title_token = "<title>"
        self.end_token = "</s>"

    def create_ngrams(self, tokens: List[str]) -> list[tuple[str, ...]]:
        """Создание n-грамм с учётом начала и конца."""
        tokens = [self.start_token] + tokens + [self.end_token]
        return [tuple(tokens[i:i + self.n_gram]) for i in range(len(tokens) - self.n_gram + 1)]

    def train(self, pairs: List[Tuple[str, str]]) -> None:
        """Обучение на списке пар (text, title)"""
        for text, title in pairs:
            text_tokens = tokenize(text)
            title_tokens = tokenize(title)
            combined = text_tokens + [self.title_token] + title_tokens
            self.vocab.update(combined)

            ngrams_list = self.create_ngrams(combined)
            for ngram in ngrams_list:
                context = ngram[:-1]
                next_word = ngram[-1]
                self.ngrams[context][next_word] += 1

    def train_from_csv(self, csv_path: str, text_col: str = "text", title_col: str = "title",
                       limit: int | None = None):
        """Обучение модели по CSV-файлу"""
        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Обучение модели"):
            text = getattr(row, text_col, None)
            title = getattr(row, title_col, None)
            if text and title:
                self.train([(text, title)])

        print(f"Обучение завершено. Количество уникальных контекстов: {len(self.ngrams)}")

    def get_probability(self, context: tuple, word: str) -> float:
        """Вычисляет вероятность по формуле P(w_t | context) = C(context + word) / C(context)"""
        context_counts = self.ngrams.get(context, {})
        total = sum(context_counts.values())
        return context_counts[word] / total if total else 0.0

    def generate_title(self, input_text: str, max_words: int = 7) -> str:
        """Генерация названия на основе входного текста. Если прямого контекста нет — выбирается ближайший по совпадению слов.
        """
        tokens = tokenize(input_text)
        tokens += [self.title_token]
        context = tuple(tokens[-(self.n_gram - 1):])

        if context not in self.ngrams:
            title_contexts = [c for c in self.ngrams.keys() if self.title_token in c]
            best_context = None
            best_score = 0

            for c in title_contexts:
                score = len(set(c) & set(context))
                if score > best_score:
                    best_score = score
                    best_context = c

            if best_context:
                context = best_context
            else:
                if title_contexts:
                    context = random.choice(title_contexts)
                else:
                    return "Без названия"

        result = []

        for _ in range(max_words):
            next_candidates = self.ngrams.get(context)
            if not next_candidates:
                break

            words = list(next_candidates.keys())
            probs = [self.get_probability(context, w) for w in words]
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]

            next_word = random.choices(words, weights=probs)[0]
            if next_word in {self.end_token, self.title_token}:
                break

            result.append(next_word)
            context = (*context[1:], next_word)

        title = " ".join(result).capitalize()
        return title if title else "Без названия"

    def evaluate_meteor(self, test_pairs: List[Tuple[str, str]]) -> float:
        """Вычисление средней METEOR-оценки для тестового набора (text, reference_title)."""
        scores = []
        for text, true_title in tqdm(test_pairs, desc="Оценка METEOR"):
            generated = self.generate_title(text)
            score = meteor_score([tokenize(true_title)], tokenize(generated))
            scores.append(score)
        return sum(scores) / len(scores) if scores else 0.0


train_df = "../../data/training_data/train_df.csv"
val_df = pd.read_csv("../../data/training_data/val_df.csv")

model = TitleNgramModel(n_gram=4)
model.train_from_csv(train_df)

val_pairs = list(zip(val_df.text, val_df.title))
meteor = model.evaluate_meteor(val_pairs)
print(f"\nСредняя METEOR-оценка на валидационном наборе: {meteor:.4f}")  # на 4-граммах получился meteor 0.0091
