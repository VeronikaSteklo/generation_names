import sys
import os

from my_models.Seq2seq.data.vocab import simple_tokenize
from my_models.Seq2seq.model.utils import load_model, generate_response

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

model, vocab = load_model(model="../../models/seq2seq/best_model_seq2seq_finetuned_2.pt")

print("\nВведите текст для генерации заголовка.")
print("Для выхода введите: exit или quit")

while True:
    try:
        user_input = input("\n>>> Введите текст: ").strip()

        if user_input.lower() in ['exit', 'quit', 'выход']:
            break

        if not user_input:
            print("Пожалуйста, введите текст.")
            continue

        tokens = simple_tokenize(user_input)
        debug_info = []

        response = generate_response(model, vocab, user_input)

        if response:
            print(f"\nОтвет: {response}")
        else:
            print("\nОтвет: [не удалось сгенерировать]")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\nОшибка: {e}")
