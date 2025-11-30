#%% md
# # Импорт библиотек
#%%
import json
import re

import pandas as pd

import matplotlib.pyplot as plt
from pandas import DataFrame

from wordcloud import WordCloud

#%% md
# # Объединим все данные в один датасет
#%%
def open_json(input_file: str) -> dict:
    with open(input_file) as json_file:
        data = json.load(json_file)
    return data
#%%
def json_to_csv_lp(json_data: dict) -> DataFrame:
    rows = []
    for category, works in json_data.items():
        for author, work_data in works.items():
            rows.append({
                "title": work_data.get("title", ""),
                "text": work_data.get("text", "")
            })

    df = pd.DataFrame(rows)
    return df
#%%
def json_to_csv(json_data: dict) -> DataFrame:
    rows = []
    for title, text in json_data.items():
        rows.append({
            "title": title,
            "text": text
        })

    df = pd.DataFrame(rows)
    return df
#%% md
# clean_data
#%%
briefly = open_json("../data/temp_data/json/briefly.json")
litprichal = open_json("../data/temp_data/json/data_litprichal.json")
proza_ru = open_json("../data/temp_data/json/data_proza_ru.json")
litres = open_json("../data/temp_data/json/litres.json")
#%%
briefly = json_to_csv(briefly)
litprichal = json_to_csv_lp(litprichal)
proza_ru = json_to_csv_lp(proza_ru)
litres = json_to_csv(litres)
#%%
data = pd.concat([briefly, litprichal, proza_ru, litres], ignore_index=True)
#%%
data
#%%
len(briefly) + len(litprichal) + len(proza_ru) + len(litres) - len(data)
#%% md
# Никакие данные не потерялись
#%% md
# # Работа с данными
#%%
data = data.apply(lambda col: col.str.lower().str.strip())
#%% md
# ## Почистим данные от всего лишнего
#%% md
# ### Удалим лишние символы
#%% md
# Удалим пропущенные значения
#%%
data.isna().sum()
#%%
data = data.dropna().reset_index(drop=True)
#%%
def clean_text(text):
    """Более аккуратная очистка текста для генерации заголовков"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s,.!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
#%%
data.text = data.text.apply(clean_text)
#%% md
# ## Уберем из названий нумерацию (том, эпизод и т.п.)
#%%
def clean_title_completely(title):
    """
    Полная очистка названия от всех видов нумерации
    """
    if pd.isna(title):
        return title

    cleaned = str(title).lower().strip()

    complex_patterns = [
        r'\(?\s*\d+\s+(?:эпизод|серия|глава|часть)\s+\d+\s+(?:том|книга|т\.)\s*\d*\s*\.?\)?',
        r'\(?\s*\d+\s+(?:том|книга|т\.)\s+\d+\s+(?:эпизод|серия|глава|часть)\s*\d*\s*\.?\)?',
        r'\b\d+\s+\d+\s+(?:том|часть|книга|эпизод|серия)\b',
        r'\b(?:том|часть|книга|эпизод|серия)\s+\d+\s+\d+\b',
        r'\b\d+[-\.,]\s*\d+\s+(?:том|часть|книга)',
        r'\b(?:том|часть|книга)\s+\d+[-\.,]\s*\d+\b',
    ]

    for pattern in complex_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    basic_patterns = [
        r'\s*(?:том|часть|книга|т\.|vol\.?|эпизод|серия|глава|выпуск)\s*[ivxlcdm0-9]+',
        r'\s*[ivxlcdm0-9]+\s*(?:том|часть|книга|т\.|vol\.?|эпизод|серия|глава|выпуск)',
        r'\s*\d+[-\.,]?\s*(?:том|часть|книга|глава)',
        r'\s*(?:том|часть|книга|глава)[-\.,]?\s*\d+',
    ]
    for pattern in basic_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    number_words = [
        "первая", "вторая", "третья", "четвертая", "пятая", "шестая", "седьмая",
        "восьмая", "девятая", "десятая", "одиннадцатая", "двенадцатая", "тринадцатая",
        "четырнадцатая", "пятнадцатая", "шестнадцатая", "семнадцатая", "восемнадцатая",
        "девятнадцатая", "двадцатая"
    ]
    text_num_pattern = r'\b(?:глава|часть|том|эпизод|серия|книга|выпуск)\s+(?:' + "|".join(number_words) + r')\b'
    cleaned = re.sub(text_num_pattern, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(
        r'\b\d+[-–]?(?:я|й|е|ой|ая|ое|ые|ых)?\s+(?:глава|часть|том|книга|серия|эпизод|выпуск)\b',
        '', cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r'\b(?:глава|часть|том|книга|серия|эпизод|выпуск)\s+\d+[-–]?(?:я|й|е|ая|ое|ые|ых)?\b',
        '', cleaned, flags=re.IGNORECASE
    )

    cleaned = re.sub(r'№', '', cleaned)

    cleaned = re.sub(r'\([^)]*\d+[^)]*\)', '', cleaned)
    cleaned = re.sub(r'\s+\d+\s*\.?$', '', cleaned)
    cleaned = re.sub(r'^\d+\s+', '', cleaned)

    cleaned = re.sub(r'[^\w\s.,!?-]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip(' ,.-')

    return cleaned
#%%
data["cleaned_title"] = data.title.apply(clean_title_completely)
#%%
changed_count = (data.title != data.cleaned_title).sum()
print(f"Изменено названий: {changed_count} из {len(data)}")
#%%
data.describe()
#%% md
# почему то есть полностью пустые тексты
#%%
data.isna().sum()
#%%
data = data[data.text != ""].reset_index(drop=True)
data = data[data.title != ""].reset_index(drop=True)
#%%
data.describe()
#%% md
# есть пустые тексты с заглушкой «описание отсутствует», удалим их
#%%
data = data[data.text != "описание отсутствует"].reset_index(drop=True)
#%%
data.describe()
#%%
data.cleaned_title.value_counts().head(20)
#%% md
# Почистим строки, в которых есть не русские буквы
#%%
def keep_only_russian(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^А-Яа-яЁё0-9\s.,!?-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

data.text = data.text.apply(keep_only_russian)
data.cleaned_title = data.cleaned_title.apply(keep_only_russian)

data = data[(data.text != "") & (data.cleaned_title != "")].reset_index(drop=True)

#%% md
# Заметим, что в названиях часто встречаются пустые названия или состоящие только из символов.
#%%
data = data[data.cleaned_title.str.strip() != ""]
noise_titles = ["***"]
data = data[~data.cleaned_title.isin(noise_titles)].reset_index(drop=True)

print(data.cleaned_title.value_counts().head(20))
print(f"Всего примеров после очистки: {len(data)}")
#%% md
# ## Почистим строки, в которых нет текста и удалим лидирующую пунктуацию
#%%
def is_meaningful(text):
    return bool(re.search(r"[А-Яа-яA-Za-z0-9]", text))

def clean_leading_punct(text):
    return re.sub(r"^[^\wА-Яа-я0-9]+", "", text).strip()
#%%
data = data[data.text.apply(is_meaningful)].reset_index(drop=True)
data.text = data.text.str.replace(r"[^\w\s,.!?-]", " ", regex=True)
data.text = data.text.str.replace(r"\s+", " ", regex=True).str.strip()
data.describe()
#%% md
# ## Посмотрим на дубликаты
#%%
data.duplicated().sum()
#%%
data.text.duplicated().sum(), data.title.duplicated().sum()
#%% md
# Заметим, что есть дубликаты в названиях, но это нормально, так как парсились разные сайты, на которых могли быть одни и те же произведения. Такое можно оставить.
# 
# Но есть дубликаты в тексте, причем большинство из них такие, что текст одинаковый, а названия разные. Такое нужно удалять, это создаст лишний шум для модели.
#%%
data = data[~data.text.duplicated(keep=False)].reset_index(drop=True)
#%%
data.duplicated().sum()
#%%
data.describe()
#%%
data.drop(columns=["title"], inplace=True)
#%%
data.rename(columns={"cleaned_title": "title"}, inplace=True)
#%%
data.to_csv("../data/temp_data/all_data_cleaned.csv", index=False)
#%% md
# # Анализ длины слов в названиях
#%%
title_counts = data.title.value_counts()
title_counts.head(10)
#%%
data["title_len"] = data.title.apply(lambda x: len(str(x).split()))

length_counts = data.groupby('title_len').size()

plt.figure(figsize=(12,6))
length_counts.plot(kind='bar')
plt.xlabel("Длина названия (слов)")
plt.ylabel("Количество названий")
plt.title("Распределение длин названий в датасете без аугментации")
plt.show()
#%% md
# Заметно, что есть названия с большим количеством слов. Это внесет в модель шум, поэтому стоит удалить их.
#%%
data.describe(include="O")
#%%
data = data[data.title_len <= 10].reset_index(drop=True)
#%%
data.describe(include="O")
#%%
data['text_len'] = data.text.str.split().apply(len)
#%%
print("Всего примеров:", len(data))
print("Уникальные заголовки:", data.title.nunique())
print("Средняя длина текста:", data.text_len.mean())
print("Средняя длина заголовка:", data.title_len.mean())
#%%
data.head()
#%%
data.drop(columns=["title_len", "text_len"], inplace=True)
#%%
data.to_csv("../data/all_data.csv", index=False)
#%% md
# # Посмотрим на облако слов
#%%
all_titles = " ".join(data.title.astype(str))

wordcloud = WordCloud(
    width=1200, height=600,
    background_color="white",
    max_words=400,
    colormap="viridis",
    collocations=False,
).generate(all_titles)
#%%
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Облако слов по заголовкам", fontsize=16)
plt.show()
#%% md
# Заметно, что самые популярные слова — предлоги, союзы и т.п., что логично. Их можно убрать, но оставим, чтобы модель училась генерировать названия, приближенные к реальности.