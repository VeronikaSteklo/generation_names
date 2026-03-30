import torch
import config
from model.model import TitleRNN
from model.utils import generate_title


def run_inference():
    # 1. Загрузка строго по твоему формату
    try:
        # weights_only=False обязателен, чтобы корректно подгрузить объект Vocab
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)

        vocab = checkpoint['vocab']
        model_state = checkpoint['model_state']

        # 2. Инициализация модели
        model = TitleRNN(
            vocab_size=len(vocab.itos),
            emb_dim=config.EMBED_DIM,
            hid_dim=config.HID_DIM,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        # 3. Загрузка весов
        model.load_state_dict(model_state)
        model.eval()

        print(f"--- Модель загружена (Vocab size: {len(vocab.itos)}) ---")
    except FileNotFoundError:
        print(f"Файл {config.MODEL_SAVE_PATH} не найден!")
        return
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return

    # 4. Цикл ввода
    print("\nВведите текст лекции для генерации заголовка (выход: 'quit')")
    while True:
        try:
            text = input("\nТекст >>> ").strip()

            if text.lower() in ['quit', 'exit', 'выход']:
                break

            if not text:
                continue

            # Генерация (используем твою функцию из utils)
            # Проверь порядок аргументов в своей generate_title!
            # Обычно это (model, text, vocab, device)
            res = generate_title(model, text, vocab, config.DEVICE)

            print(f"Заголовок: {res}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    run_inference()