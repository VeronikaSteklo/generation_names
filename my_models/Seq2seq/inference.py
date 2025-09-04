from __future__ import annotations

import re
import torch

from .config import (
    BEST_MODEL_PATH, SRC_VOCAB_PATH, TRG_VOCAB_PATH,
    HID_DIM, ENC_EMB_DIM, DEC_EMB_DIM, ENC_DROPOUT, DEC_DROPOUT, N_LAYERS
)
from .data import Vocab, normalize_text
from .model import EncoderLSTM, Decoder, Seq2Seq


def generate_title(model: Seq2Seq, input_text: str, input_vocab: Vocab, target_vocab: Vocab, max_len: int = 50, device: str | torch.device = "cpu", temperature: float = 0.7) -> str:
    model.eval()
    tokens = re.findall(r"\w+|[.,!?;]", input_text.lower())
    src_idxs = [input_vocab.word2index.get(t, input_vocab.word2index["<unk>"]) for t in tokens]
    if not src_idxs:
        return "Невозможно проанализировать текст"
    src_tensor = torch.LongTensor(src_idxs).unsqueeze(0).to(device)

    trg_idxs = [target_vocab.word2index["<sos>"]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_idxs).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        logits = output[0, -1]
        probs = (logits / max(1e-6, temperature)).softmax(dim=-1)
        pred = torch.multinomial(probs, 1).item()
        if pred == target_vocab.word2index["<eos>"] or (i > 10 and len(set(trg_idxs[-5:])) < 2):
            break
        trg_idxs.append(pred)

    words = []
    for idx in trg_idxs[1:]:
        w = target_vocab.index_to_word(idx)
        if w not in ["<pad>", "<unk>", "<sos>", "<eos>"] and not w.isdigit():
            words.append(w)
    result = ' '.join(words).capitalize()
    result = re.sub(r'\s([?.!,](?:\s|$))', r'\1', result)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_vocab = Vocab.load(SRC_VOCAB_PATH)
    target_vocab = Vocab.load(TRG_VOCAB_PATH)

    encoder = EncoderLSTM(input_vocab.n_words, HID_DIM, dropout_p=ENC_DROPOUT)
    decoder = Decoder(target_vocab.n_words, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("Генератор названий\nВведите текст (или 'выход' для завершения):")
    while True:
        inp = input("\n> ")
        if inp.strip().lower() in {"выход", "exit", "quit"}:
            break
        if not inp.strip():
            print("Пожалуйста, введите текст.")
            continue

        norm_tokens = normalize_text(inp)
        inp_for_model = ' '.join(norm_tokens) if norm_tokens else inp

        title = generate_title(model, inp_for_model, input_vocab, target_vocab, device=device)
        print("\nСгенерированное название:\n" + title)


if __name__ == "__main__":
    main()