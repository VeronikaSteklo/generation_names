import torch
import config
from model.model import TitleRNN
from model.utils import generate_title


def run_inference():
    try:
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)

        vocab = checkpoint['vocab']
        model_state = checkpoint['model_state']

        model = TitleRNN(
            vocab_size=len(vocab.itos),
            emb_dim=config.EMBED_DIM,
            hid_dim=config.HID_DIM,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        ).to(config.DEVICE)

        model.load_state_dict(model_state)
        model.eval()

        print(f"--- Модель загружена (Vocab size: {len(vocab.itos)}) ---")
    except FileNotFoundError:
        print(f"Файл {config.MODEL_SAVE_PATH} не найден!")
        return
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
        return

    print("\nВведите текст лекции для генерации заголовка (выход: 'quit')")
    while True:
        try:
            text = input("\nТекст >>> ").strip()

            if text.lower() in ['quit', 'exit', 'выход']:
                break

            if not text:
                continue

            res = generate_title(model, text, vocab, config.DEVICE)

            print(f"Заголовок: {res}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == "__main__":
    run_inference()