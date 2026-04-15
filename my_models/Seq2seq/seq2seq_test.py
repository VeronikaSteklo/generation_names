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
        unk_count = 0

        for t in tokens:
            if t in vocab.word2index:
                debug_info.append(f"{t}")
            else:
                debug_info.append(f"[{t} -> UNK]")
                unk_count += 1

        print("-" * 30)
        print(f"Как модель видит вход: {''.join(debug_info)}")
        if unk_count > 0:
            print(f"Внимание: {unk_count} слов(а) заменены на <unk>")
        print("-" * 30)

        response = generate_response(model, vocab, user_input)

        if response:
            print(f"\nОтвет: {response}")
        else:
            print("\nОтвет: [не удалось сгенерировать]")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\nОшибка: {e}")
