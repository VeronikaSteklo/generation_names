import math
import os
import pickle

from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import re

from tqdm import tqdm

from my_models.history.FNNLM.config import *
from my_models.history.FNNLM.data.dataset import FNNLMDataset
from my_models.history.FNNLM.model.model import FNNLM


def train_model():
    train_dataset = FNNLMDataset(TRAIN_DATASET, CONTEXT_SIZE)
    val_dataset = FNNLMDataset(VAL_DATASET, CONTEXT_SIZE, vocab=train_dataset.vocab, first_sentences=2)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FNNLM(len(train_dataset.vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM, DROPOUT).to(DEVICE)
    pad_id = train_dataset.vocab["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    counter = 0
    best_val_loss = float('inf')

    model.train()
    pbar_epochs = tqdm(range(EPOCHS), desc="Обучение FNNLM")

    for epoch in pbar_epochs:
        model.train()
        train_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- ВАЛИДАЦИЯ ---
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        avg_val_loss = val_loss / val_samples
        val_ppl = math.exp(avg_val_loss)

        pbar_epochs.set_postfix({
            "tr_loss": f"{avg_train_loss:.3f}",
            "val_loss": f"{avg_val_loss:.3f}",
            "val_PPL": f"{val_ppl:.1f}"
        })

        # --- ЛОГИКА РАННЕЙ ОСТАНОВКИ И СОХРАНЕНИЯ ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\nРанняя остановка на эпохе {epoch + 1}. Лучший Val Loss: {best_val_loss:.4f}")
                break

    dataset_info = {
        'vocab': train_dataset.vocab,
        'rev_vocab': train_dataset.rev_vocab,
        'vocab_size': len(train_dataset.vocab)
    }
    with open('../../../models/fnnlm/dataset_info.pkl', 'wb') as f:
        pickle.dump(dataset_info, f)

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    return model, train_dataset


def generate_title(model, input_text, vocab, rev_vocab, max_words=12, temperature=0.7, top_k=5):
    device = next(model.parameters()).device
    model.eval()

    clean_input = re.sub(r'[^\w\s]', '', input_text.lower()).split()
    if len(clean_input) > CONTEXT_SIZE:
        current_context = clean_input[-CONTEXT_SIZE:]
    else:
        current_context = ['<PAD>'] * (CONTEXT_SIZE - len(clean_input)) + clean_input

    result_title = []

    for _ in range(max_words):
        input_ids = torch.tensor([[vocab.get(w, vocab['<UNK>']) for w in current_context]]).to(device)

        with torch.no_grad():
            logits = model(input_ids)

            logits /= temperature

            top_logits, top_indices = torch.topk(logits, top_k)

            probs = F.softmax(top_logits, dim=-1)

            idx_in_top = torch.multinomial(probs, num_samples=1).item()
            pred_id = top_indices[0, idx_in_top].item()

            pred_word = rev_vocab[pred_id]

        if pred_word == '<END>':
            break

        if pred_word not in ['<PAD>', '<UNK>']:
            result_title.append(pred_word)

        current_context = current_context[1:] + [pred_word]

    return " ".join(result_title)