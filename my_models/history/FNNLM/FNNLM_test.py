import torch
import re
import pickle
from my_models.history.FNNLM.model.model import FNNLM
from my_models.history.FNNLM.config import *
from my_models.history.FNNLM.model.utils import generate_title


def load_model_with_dataset():
    """Загрузка модели с информацией о датасете"""

    try:
        dataset_path = "../../../models/fnnlm/dataset_info.pkl"
        with open(dataset_path, 'rb') as f:
            dataset_info = pickle.load(f)

        vocab = dataset_info['vocab']
        rev_vocab = dataset_info['rev_vocab']
        vocab_size = dataset_info['vocab_size']

        print(f"Информация о датасете загружена!")
        print(f"Размер словаря: {vocab_size}")

        model = FNNLM(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)

        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        model.eval()

        print("Модель успешно загружена!")
        return model, vocab, rev_vocab

    except FileNotFoundError:
        print("Файл с информацией о датасете не найден!")
        print("Убедитесь, что вы сохранили dataset_info.pkl при обучении")
        return None, None, None
    except Exception as e:
        print(f"Ошибка загрузки: {str(e)}")
        return None, None, None



def interactive_generator():
    model, vocab, rev_vocab = load_model_with_dataset()
    if model is None: return

    print("\nГотово! Введите текст. Для выхода — 'quit'.")

    while True:
        user_text = input(">>> ").strip()
        if user_text.lower() == 'quit': break
        if len(user_text) < 5: continue

        try:
            title = generate_title(model, user_text, vocab, rev_vocab)
            if title.strip() == "":
                print(f"Модель ничего не сгенерировала\n")
                continue
            print(f"Заголовок: {title}\n")
        except Exception as e:
            print(f"Ошибка при генерации: {e}")


if __name__ == "__main__":
    interactive_generator()
