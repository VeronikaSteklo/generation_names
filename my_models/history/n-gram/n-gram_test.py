from n_gram_model import TitleNgramModel

model = TitleNgramModel.load("../../models/history/title_ngram_model.pkl")

while True:
    text = input("> Введите текст или exit для выхода")
    if text.lower() == "exit":
        break
    print(f"Сгенерированное название: {model.generate_title(text)}")
