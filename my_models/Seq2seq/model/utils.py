import os
import random

import numpy as np
from tqdm import tqdm

from my_models.Seq2seq.config import *
from my_models.Seq2seq.data.vocab import Vocab, simple_tokenize
from my_models.Seq2seq.model.decoder import Decoder
from my_models.Seq2seq.model.encoder import Encoder
from my_models.Seq2seq.model.seq2seq import Seq2Seq


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(dataloader, desc=" Batch", leave=False)

    for src, trg in pbar:
        src, trg = src.to(model.device), trg.to(model.device)
        optimizer.zero_grad()

        outputs = model(src, trg)

        out_dim = outputs.shape[-1]

        outputs = outputs[:, 1:].reshape(-1, out_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(model.device), trg.to(model.device)

            outputs = model(src, trg, teacher_forcing_ratio=0.0)

            out_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].reshape(-1, out_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def generate_response(model, vocab, text, beam_width=3, max_len=MAX_TRG_LEN):
    model.eval()

    sos_idx = vocab.word2index["<sos>"]
    eos_idx = vocab.word2index["<eos>"]

    with torch.no_grad():
        tokens = simple_tokenize(str(text))
        src_indices = [vocab.word2index.get(t, 1) for t in tokens[:MAX_SRC_LEN]]
        src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)

        encoder_outputs, hidden, cell = model.encoder(src_tensor)

        beams = [(0.0, [sos_idx], hidden, cell)]

    for _ in range(max_len):
        new_beams = []

        for score, tokens, h, c in beams:
            if tokens[-1] == eos_idx:
                new_beams.append((score, tokens, h, c))
                continue

            input_tok = torch.tensor([tokens[-1]]).to(device)

            output, new_h, new_c = model.decoder(input_tok, h, c, encoder_outputs)

            log_probs = torch.log_softmax(output, dim=1)
            top_v, top_i = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                next_score = score + top_v[0, i].item()
                next_tokens = tokens + [top_i[0, i].item()]
                new_beams.append((next_score, next_tokens, new_h, new_c))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        if all(b[1][-1] == eos_idx for b in beams):
            break

    best_tokens = beams[0][1]

    result_words = []
    for idx in best_tokens:
        if idx == eos_idx:
            break
        if idx != sos_idx:
            result_words.append(vocab.index2word.get(idx, "<unk>"))

    return "".join(result_words)


def load_model(vocab_path=VOCAB_PATH, model=MODEL_SAVE_PATH):
    """Загружает обученную модель и словарь"""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Файл словаря не найден: {vocab_path}")
    vocab = Vocab.load(vocab_path)

    if not os.path.exists(model):
        raise FileNotFoundError(f"Файл модели не найден: {model}")
    
    checkpoint = torch.load(model, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_vocab_size = checkpoint.get('vocab_size', checkpoint['model_state_dict']['encoder.embedding.weight'].shape[0])
        model_state = checkpoint['model_state_dict']
        meta_info = f"epoch={checkpoint.get('epoch', '?')}, val_loss={checkpoint.get('val_loss', 0):.4f}"
    else:
        model_vocab_size = checkpoint['encoder.embedding.weight'].shape[0]
        model_state = checkpoint
        meta_info = "старый формат"
    
    if model_vocab_size != vocab.n_words:
        print(f"ВНИМАНИЕ: несоответствие размеров словаря.")
        print(f"  В файле vocab.pkl: {vocab.n_words}")
        print(f"  В checkpoint модели: {model_vocab_size}")
        print(f"  Используем {model_vocab_size} для инициализации модели.")
        model_vocab_size = model_vocab_size
    else:
        print(f"Размер словаря совпадает: {model_vocab_size}")

    encoder = Encoder(model_vocab_size, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT)
    decoder = Decoder(model_vocab_size, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(model_state)
    model.eval()
    print(f"Модель загруена успешно! ({meta_info})")

    return model, vocab
