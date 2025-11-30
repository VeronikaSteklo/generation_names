import pandas as pd
from n_gram_model import TitleNgramModel

train_df = "../../data/training_data/train_df.csv"
val_df = pd.read_csv("../../data/training_data/val_df.csv")

model = TitleNgramModel(n_gram=3)
model.train_from_csv(train_df)
model.save(path="../../models/history/title_ngram_model.pkl")

val_pairs = list(zip(val_df.text, val_df.title))
meteor = model.evaluate_meteor(val_pairs)
print(f"\nСредняя METEOR-оценка на валидационном наборе: {meteor:.4f}")