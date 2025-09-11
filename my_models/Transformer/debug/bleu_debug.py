import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from my_models.Transformer.model.utils import generate_title


def debug_bleu(model, tokenizer, data_loader, device):
    """Упрощенная проверка BLEU"""
    refs = []
    hyps = []
    exact_matches = 0
    token_matches = 0

    model.eval()
    with torch.no_grad():
        batch_ = 0
        for batch in data_loader:
            for text, true_title in zip(batch["src_text"], batch["tgt_text"]):
                gen_title = generate_title(model, tokenizer, text, device=device)

                true_tokens = tokenizer.encode(true_title, add_special_tokens=False)
                gen_tokens = tokenizer.encode(gen_title, add_special_tokens=False)

                refs.append([true_title.split()])
                hyps.append(gen_title.split())

                if true_title == gen_title:
                    exact_matches += 1

                if true_tokens == gen_tokens:
                    token_matches += 1

                batch_ += 1

                print(f"Верный заголовок: '{true_title}'")
                print(f"Сгенерированный заголовок: '{gen_title}'")
                print(f"Количество совпадающих токенов: {true_tokens == gen_tokens}")
                print("---")
                if batch_ == 1:
                    break
            break

    smooth_fn = SmoothingFunction().method1
    bleu_score = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)

    print(f"Точные совпадения: {exact_matches}/{len(refs)}")
    print(f"Совпадения токенов: {token_matches}/{len(refs)}")
    print(f"BLEU score: {bleu_score}")

    return bleu_score
