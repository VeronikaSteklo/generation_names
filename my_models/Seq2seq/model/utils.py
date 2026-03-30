import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0.0

    for src, trg in dataloader:
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
