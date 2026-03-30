import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

# Твои конфиги и модули
import config
from data.vocab import Vocab
from data.dataset import TitleDataset, collate_fn
from model.encoder import Encoder
from model.decoder import Decoder
from model.seq2seq import Seq2Seq
from my_models.Seq2seq.model.utils import train_epoch, evaluate


train_df = pd.read_csv(config.TRAIN_DATA_PATH)
train_df = train_df.groupby('title').head(config.N_SAMPLES).reset_index(drop=True).head(config.LIMIT)
val_df = pd.read_csv(config.VAL_DATA_PATH).head(config.LIMIT)
print(f"Количество объектов в train: {len(train_df)}, val: {len(val_df)}")

vocab = Vocab()
vocab.build_vocab(train_df['text'].values, train_df['title'].values)
vocab.save(config.VOCAB_PATH)

train_dataset = TitleDataset(train_df, vocab, config.MAX_SRC_LEN, config.MAX_TRG_LEN)
val_dataset = TitleDataset(val_df, vocab, config.MAX_SRC_LEN, config.MAX_TRG_LEN)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn)

encoder = Encoder(vocab.n_words, config.ENC_EMB_DIM, config.HID_DIM, config.ENC_DROPOUT)
decoder = Decoder(vocab.n_words, config.DEC_EMB_DIM, config.HID_DIM, config.DEC_DROPOUT)
model = Seq2Seq(encoder, decoder, config.device).to(config.device)

PAD_IDX = vocab.word2index["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)

best_val_loss = float('inf')
history = []

epochs_range = range(1, config.NUM_EPOCHS + 1)
pbar = tqdm(epochs_range, desc="Training", unit="epoch")
epochs_no_improve = 0

for epoch in pbar:
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

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= config.PATIENCE:
        print(f"Ранняя остановка на {epoch:02} эпохе | Best val loss: {best_val_loss:.4f}")
        break


    pbar.set_postfix({
        'Train_loss': f"{train_loss:.3f}",
        'Val_loss': f"{val_loss:.3f}",
        'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
    })

history_path = config.MODEL_SAVE_PATH.replace('.pt', '_history.csv')
pd.DataFrame(history).to_csv(history_path, index=False)
print(f"\nОбучение завершено. История сохранена в {history_path}")
