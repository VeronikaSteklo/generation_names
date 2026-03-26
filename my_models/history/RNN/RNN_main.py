import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial

from data.dataset import TitleDataset, Vocab, collate_fn
from model.model import TitleRNN
from model.utils import train_epoch, evaluate
import config

train_df = pd.read_csv(config.TRAIN_DATA_PATH).head(config.LIMIT)
val_df = pd.read_csv(config.VAL_DATA_PATH).head(config.LIMIT)

all_text = (train_df['text'].astype(str).tolist() +
            train_df['title'].astype(str).tolist() +
            val_df['text'].astype(str).tolist() +
            val_df['title'].astype(str).tolist())
vocab = Vocab(all_text)

train_dataset = TitleDataset(train_df, vocab)
val_dataset = TitleDataset(val_df, vocab)

pad_idx = vocab.stoi[config.PAD_TOKEN]
collate_p = partial(collate_fn, pad_idx=pad_idx)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_p)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_p)

model = TitleRNN(
    vocab_size=len(vocab.itos),
    emb_dim=config.EMBED_DIM,
    hid_dim=config.HID_DIM,
    n_layers=config.N_LAYERS,
    dropout=config.DROPOUT
).to(config.DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
print(f"Размер словаря: {len(vocab)} | Train образцов: {len(train_dataset)} | Val образцов: {len(val_dataset)}")
best_loss = float('inf')
epoch_no_improve = 0
epoch_range = tqdm(range(config.EPOCHS), desc="Training progress")
for epoch in epoch_range:
    train_loss = train_epoch(model, train_loader, optimizer, criterion, config.DEVICE)

    val_loss = evaluate(model, val_loader, criterion, config.DEVICE)

    epoch_range.set_postfix({
        'train_loss': f'{train_loss:.4f}',
        'val_loss': f'{val_loss:.4f}'
    })

    if val_loss < best_loss:
        best_loss = val_loss
        epoch_no_improve = 0
        torch.save({'model_state': model.state_dict(), 'vocab': vocab}, config.MODEL_SAVE_PATH)
    else:
        epoch_no_improve += 1
    if epoch_no_improve == config.PATIENT:
        print(f"Ранняя остановка на {epoch + 1} эпохе | Best Loss: {best_loss:.4f}")
        break
