import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

import config
from data.vocab import Vocab
from data.dataset import TitleDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from my_models.Seq2seq.model.utils import train_epoch, evaluate

# import os
# os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

full_train = pd.read_csv(config.TRAIN_DATA_PATH)
vocab = Vocab()
vocab.build_vocab(full_train['text'].values, full_train['title'].values)
vocab.save(config.VOCAB_PATH)

train_df = pd.read_csv(config.TRAIN_DATA_PATH)

train_df = train_df.groupby('title').head(config.N_SAMPLES)

train_df = train_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

train_df = train_df.head(config.LIMIT)
val_df = pd.read_csv(config.VAL_DATA_PATH).sample(frac=1, random_state=config.SEED).head(int(len(train_df) * 0.4))
print(f"Количество объектов в train: {len(train_df)}, val: {len(val_df)}")

train_dataset = TitleDataset(train_df, vocab, config.MAX_SRC_LEN, config.MAX_TRG_LEN)
val_dataset = TitleDataset(val_df, vocab, config.MAX_SRC_LEN, config.MAX_TRG_LEN)

del train_df
del val_df
del full_train

import gc
gc.collect()

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

encoder = Encoder(vocab.n_words, config.ENC_EMB_DIM, config.HID_DIM, config.ENC_DROPOUT)
decoder = Decoder(vocab.n_words, config.DEC_EMB_DIM, config.HID_DIM, config.DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, config.device).to(config.device)

PAD_IDX = vocab.word2index["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.0)
optimizer = optim.AdamW(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

best_val_loss = float('inf')
history = []

epochs_range = range(1, config.NUM_EPOCHS + 1)
# pbar = tqdm(epochs_range, desc="Training", unit="epoch")
epochs_no_improve = 0
if torch.backends.mps.is_available():
    torch.mps.empty_cache()

for epoch in epochs_range:
    start_time = time.time()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, config.CLIP)
    val_loss = evaluate(model, val_loader, criterion)

    scheduler.step(val_loss)

    history.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': optimizer.param_groups[0]['lr']
    })
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab.n_words,
        'epoch': epoch,
        'val_loss': val_loss,
        'train_loss': train_loss,
    }
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(checkpoint, config.MODEL_SAVE_PATH)
        print(f"\nЛучшая модель сохранена (vocab_size={vocab.n_words}, epoch={epoch:02d}, val_loss={val_loss:.4f})")
    else:
        epochs_no_improve += 1

    torch.save(checkpoint, config.MODEL_SAVE_PATH.replace(".pt", "_overfitting.pt"))

    print(f"epoch: {epoch:02d}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}, time: {time.time() - start_time:.2f}")

    if epochs_no_improve >= config.PATIENCE:
        print(f"Ранняя остановка на {epoch:02} эпохе | Best val loss: {best_val_loss:.4f}")
        break

history_path = config.MODEL_SAVE_PATH.replace('.pt', '_history.csv')
pd.DataFrame(history).to_csv(history_path, index=False)
print(f"\nОбучение завершено. История сохранена в {history_path}")
