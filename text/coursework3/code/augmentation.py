# %% md
# # Импорт библиотек
# %%
import re

import pandas as pd
import nltk
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import numpy as np

from tqdm import tqdm

nltk.download("punkt")

from nltk.tokenize import sent_tokenize

# %%
data = pd.read_csv("../data/all_data.csv")
# %%
data.describe()
# %% md
# # Разделение данных
# %%
train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
# %%
len(train_df), len(val_df)
# %%
val_df.describe()


# %% md
# Заметим, что в валидацию попали несколько текстов с одинаковыми названиями. Это не очень хорошо, так как может исказить результаты оценки модели.
# 
# Переделаем разделение.
# %%
def split_with_controlled_test_size(data, target_test_size=0.2, random_state=42):
    np.random.seed(random_state)

    title_groups = data.groupby('title').apply(lambda x: x.index.tolist()).to_dict()

    unique_titles = list(title_groups.keys())
    np.random.shuffle(unique_titles)

    train_indices = []
    test_indices = []

    target_test_count = int(len(data) * target_test_size)

    for title in unique_titles:
        indices = title_groups[title]

        if len(test_indices) < target_test_count:
            test_idx = np.random.choice(indices, 1)[0]
            test_indices.append(test_idx)
            train_indices.extend([idx for idx in indices if idx != test_idx])
        else:
            train_indices.extend(indices)

    return data.iloc[train_indices], data.iloc[test_indices]


train_df, val_df = split_with_controlled_test_size(data)
# %%
print(f"Общий размер данных: {len(data)}")
print(f"Тренировочная выборка: {len(train_df)} записей ({len(train_df) / len(data) * 100:.1f}%)")
print(f"Валидационная выборка: {len(val_df)} записей ({len(val_df) / len(data) * 100:.1f}%)")
# %%
val_df.describe()
# %% md
# Теперь в валидации нет повторяющихся заголовков.
# %%
val_df.to_csv("../data/training_data/val_df.csv", index=False)
# %% md
# val_df.to_csv("../data/val_df.csv", index=False)# Аугментация данных
# %% md
# Аугментировать будем только train набор, чтобы не произошла утечка.
# %%
train_df.reset_index(drop=True, inplace=True)


# %%
def split_text_into_chunks(text, sentences_per_chunk=3):
    sentences = sent_tokenize(text, language="russian")
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        chunks.append(chunk)
    return chunks


# %%
augmented_rows = []
for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    chunks = split_text_into_chunks(row.text, sentences_per_chunk=3)
    for chunk in chunks:
        augmented_rows.append({"text": chunk, "title": row.title})
# %%
augmented_dataset = pd.DataFrame(augmented_rows)
# %%
augmented_dataset.describe()


# %% md
# ## Почистим строки, в которых нет текста и удалим лидирующую пунктуацию
# %%
def is_meaningful(text):
    return bool(re.search(r"[А-Яа-яA-Za-z0-9]", text))


def clean_leading_punct(text):
    return re.sub(r"^[^\wА-Яа-я0-9]+", "", text).strip()


# %%
augmented_dataset = augmented_dataset[augmented_dataset.text.apply(is_meaningful)].reset_index(drop=True)
augmented_dataset.text = augmented_dataset.text.str.replace(r"[^\w\s,.!?-]", " ", regex=True)
augmented_dataset.text = augmented_dataset.text.str.replace(r"\s+", " ", regex=True).str.strip()
augmented_dataset.describe()
# %% md
# Удалим строки, с небольшим количеством данных
# %%
augmented_dataset = augmented_dataset[augmented_dataset.text.str.strip().str.len() > 10]
# %%
augmented_dataset.describe()
# %% md
# Заметим, что у нас очень много текстов с одинаковыми названиями. Это может плохо повлиять на модель, если она будет видеть одни и те же названия. Оставим только по 500 каждого
# %%
max_per_title = 500
augmented_dataset = augmented_dataset.groupby("title").head(max_per_title).reset_index(drop=True)

print(augmented_dataset.title.value_counts().head(10))
# %%
augmented_dataset.describe()
# %% md
# Появились одинаковые тексты
# %%
augmented_dataset.duplicated().sum()
# %% md
# Есть полные дубликаты текст + название. Такое удалим.
# %%
augmented_dataset.drop_duplicates(inplace=True)
# %%
augmented_dataset.reset_index(drop=True, inplace=True)
# %%
augmented_dataset.describe()
# %% md
# Все еще остались одинаковые тексты, но теперь у них разные названия. Удалим и их.
# %%
augmented_dataset = augmented_dataset[~augmented_dataset.text.duplicated(keep=False)].reset_index(drop=True)
# %%
augmented_dataset.describe()
# %%
augmented_dataset
# %%
augmented_dataset.to_csv("../data/training_data/train_df.csv", index=False)
# %% md
# ## Посмотрим на распределение названий по длине после разделения
# %%
train_df = augmented_dataset.copy()
# %%
dup_titles = (
    augmented_dataset.groupby("text")["title"]
    .nunique()
    .reset_index()
    .query("title > 1")
)

print(f"Текстов с одинаковыми содержаниями, но разными названиями: {len(dup_titles)}")
# %%
data["title_len"] = data.title.apply(lambda x: len(str(x).split()))
length_counts = data.groupby('title_len').size()
train_df["title_len"] = train_df.title.apply(lambda x: len(str(x).split()))
length_counts_aug = train_df.groupby('title_len').size()
val_df["title_len"] = val_df.title.apply(lambda x: len(str(x).split()))
length_counts_aug_val = val_df.groupby('title_len').size()
# %%
fig = go.Figure()

fig.add_trace(go.Bar(
    x=length_counts.index,
    y=length_counts.values,
    name="Без аугментации",
    marker_color="blue"
))

fig.add_trace(go.Bar(
    x=length_counts_aug.index,
    y=length_counts_aug.values,
    name="train после аугментации",
    marker_color="orange"
))

fig.add_trace(go.Bar(
    x=length_counts_aug_val.index,
    y=length_counts_aug_val.values,
    name="test после аугментации",
    marker_color="green"
))

fig.update_layout(
    title="Сравнение распределения длин названий до и после аугментации",
    xaxis_title="Длина названия (слов)",
    yaxis_title="Количество названий",
    barmode="group",
    bargap=0.2,
    bargroupgap=0.1,
    width=1000,
    height=500
)

fig.show()
