from functools import lru_cache
import re
from typing import List

import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

morph = MorphAnalyzer()

_stop_words = set(stopwords.words('russian'))
_stop_words.update({'это', 'который', 'весь', 'свой', 'такой', 'тем', 'чтобы'})

@lru_cache(maxsize=10000)
def normalize_word(word: str) -> str:
    """Нормализация слова через pymorphy3"""
    return morph.parse(word)[0].normal_form.replace('ё', 'е')

def normalization(tokens: List[str]) -> List[str]:
    """Применение нормализации ко всем токенам"""
    return [normalize_word(w) for w in tokens]

def normalize_text(text: str) -> List[str]:
    """Полная предобработка текста: очистка, токенизация, нормализация"""
    from nltk import word_tokenize
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = normalization(tokens)
    return [t for t in tokens if t not in _stop_words]