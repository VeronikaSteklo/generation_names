import math

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import TitleVocab, LSTMTitleDataset
from model.model import LSTMTitleGen
import config

train_full = pd.read_csv(config.TRAIN_DATA_PATH)
train_full = train_full.sample(frac=1, random_state=42).reset_index(drop=True)
train_df = train_full.head(config.LIMIT).dropna(subset=['text', 'title'])

val_full = pd.read_csv(config.VAL_DATA_PATH)
val_full = val_full.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_full.head(config.LIMIT).dropna(subset=['text', 'title'])

vocab = TitleVocab(train_df)

train_dataset = LSTMTitleDataset(train_df, vocab, config.MAX_LEN)
val_dataset = LSTMTitleDataset(val_df, vocab, config.MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

model = LSTMTitleGen(
    vocab_size=len(vocab),
    embed_dim=config.EMBED_DIM,
    hidden_dim=config.HIDDEN_DIM,
    num_layers=config.NUM_LAYERS,
    dropout=config.DROPOUT
).to(config.DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

epoch_no_improve = 0
best_loss = float('inf')

print(f"Размер словаря: {len(vocab)} | Train образцов: {len(train_dataset)} | Val образцов: {len(val_dataset)}")

pbar = tqdm(range(config.EPOCHS), desc="Training Progress")
for epoch in pbar:
    model.train()
    train_loss = 0
    for inputs, targets, masks in train_loader:
        inputs, targets, masks = inputs.to(config.DEVICE), targets.to(config.DEVICE), masks.to(config.DEVICE)
        optimizer.zero_grad()
        logits, _ = model(inputs)
        raw_loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))

        masked_loss = raw_loss * masks.view(-1)

        loss = masked_loss.sum() / (masks.sum() + 1e-9)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs, targets, masks = inputs.to(config.DEVICE), targets.to(config.DEVICE), masks.to(config.DEVICE)

            logits, _ = model(inputs)

            raw_loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            masked_loss = raw_loss * masks.view(-1)
            loss = masked_loss.sum() / (masks.sum() + 1e-9)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')

    pbar.set_postfix({
        'train loss': f"{avg_train_loss:.4f}",
        'val loss': f"{avg_val_loss:.4f}",
        'val ppl': f"{val_ppl:.2f}"
    })

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        epoch_no_improve = 0
        torch.save({'model_state': model.state_dict(), 'vocab': vocab}, config.MODEL_SAVE_PATH)
    else:
        epoch_no_improve += 1

    if epoch_no_improve >= config.PATIENCE:
        print(f"Ранняя остановка на {epoch + 1} эпохе | Best Loss: {best_loss:.4f}")
        break
