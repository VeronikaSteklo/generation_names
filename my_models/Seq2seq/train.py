from __future__ import annotations

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

from config import (
    DATA_CSV, SRC_VOCAB_PATH, TRG_VOCAB_PATH, BEST_MODEL_PATH,
    MAX_VOCAB_SIZE, BATCH_SIZE, NUM_EPOCHS, CLIP, EARLY_STOPPING_PATIENCE,
    LR, WEIGHT_DECAY, LABEL_SMOOTHING, HID_DIM, ENC_EMB_DIM, DEC_EMB_DIM,
    ENC_DROPOUT, DEC_DROPOUT, N_LAYERS, PAD_IDX
)
from .data import Vocab, TitleDataset, collate_fn
from .model import EncoderLSTM, Decoder, Seq2Seq


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = pd.read_csv(DATA_CSV)
    pairs = list(dataset[["title", "text"]].itertuples(index=False, name=None))
    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)

    input_vocab = Vocab("input")
    target_vocab = Vocab("target")
    for title, text in train_pairs:
        input_vocab.addText(text)
        target_vocab.addText(title)
    input_vocab.build_vocab(max_size=MAX_VOCAB_SIZE, is_text=True)
    target_vocab.build_vocab(max_size=MAX_VOCAB_SIZE, is_text=False)
    input_vocab.save(SRC_VOCAB_PATH)
    target_vocab.save(TRG_VOCAB_PATH)

    train_dataset = TitleDataset(train_pairs, input_vocab, target_vocab)
    val_dataset = TitleDataset(val_pairs, input_vocab, target_vocab)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    INPUT_DIM = input_vocab.n_words
    OUTPUT_DIM = target_vocab.n_words

    encoder = EncoderLSTM(INPUT_DIM, HID_DIM, dropout_p=ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=1, threshold=0.01
    )

    best_val_loss = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        clip=CLIP,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        model_save_path=BEST_MODEL_PATH,
    )

    print(f"Best val loss: {best_val_loss:.3f}")


if __name__ == "__main__":
    main()