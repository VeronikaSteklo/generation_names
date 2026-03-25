import torch
import config
from model.model import LSTMTitleGen
from model.utils import generate_title


def load_model_and_vocab():
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
        vocab = checkpoint['vocab']

        print(f"Размер словаря: {len(vocab)}")

        model = LSTMTitleGen(
            vocab_size=len(vocab),
            embed_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        return model, vocab

    except FileNotFoundError:
        print(f"Файл {config.MODEL_SAVE_PATH} не найден!")
        return None, None
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return None, None


model, vocab = load_model_and_vocab()

print("\nГотово! Введите текст. Для выхода — 'quit'.")

while True:
    try:
        user_text = input("\nВведите текст >>> ").strip()

        if user_text.lower() in ['quit', 'exit', 'выход']:
            print("Выход...")
            break

        if len(user_text) < 5:
            print("Слишком короткий текст, попробуйте еще раз.")
            continue

        title = generate_title(model, vocab, config.DEVICE, user_text)

        if not title or title.strip() == "":
            print("Модель ничего не сгенерировала.")
        else:
            print(f"Заголовок: {title}")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Ошибка генерации: {e}")