from transformers import T5ForConditionalGeneration, T5Tokenizer
from config import MODEL_PATH, DEVICE
from model.generation import generate_title


def run_test():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    print("Введите 'exit' или 'выход' для завершения.\n")

    while True:
        user_input = input("Введите отрывок текста для генерации названия:\n> ")

        if user_input.lower() in ['exit', 'выход', 'quit']:
            break

        if not user_input.strip():
            continue

        try:
            title = generate_title(user_input, model, tokenizer, DEVICE)

            print("\n" + "=" * 50)
            print(f"РЕЗУЛЬТАТ: {title}")
            print("=" * 50 + "\n")

        except Exception as e:
            print(f"Ошибка при генерации: {e}")


if __name__ == "__main__":
    run_test()
