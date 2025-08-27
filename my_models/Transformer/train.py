import time

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Используемое устройство: {device}")

from my_models.Transformer.data_loader import TextTitleDataset, collate_fn
from my_models.Transformer.model.model import make_model, run_epoch, generate_title, evaluate_bleu

NUM_EPOCHS = 25
PATIENCE = 2
MIN_DELTA = 2 * 1e-2
MODEL_PATH = "../../models/transformer.pt"
best_train_loss = float("inf")
best_val_loss = float("inf")
epochs_no_improve = 0

model, tokenizer = make_model(N=2, d_model=128, d_ff=256, h=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

dataset = TextTitleDataset(
    "../../data/all_data.csv",
    tokenizer=tokenizer,
    limit=100
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Количество данных в тренировочном датасете: {len(train_dataset)}")
print(f"Количество данных в тестовом датасете: {len(val_dataset)}")

pad_id = tokenizer.token_2_idx("<pad>")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_id, device=device)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, pad_id, device=device)
)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = run_epoch(
        model, tokenizer, data_loader=train_loader,
        optimizer=optimizer, device=device
    )

    val_loss = run_epoch(model, tokenizer, val_loader, device, train=False)

    if (epoch + 1) % 5 == 0:
        bleu, bleu1, bleu4 = evaluate_bleu(model, tokenizer, val_loader, device)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}, "
              f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
              f"BLEU = {bleu:.4f}, BLEU1 = {bleu1:.4f}, BLEU4 = {bleu4:.4f}, "
              f"time = {epoch_time:.2f} sec")
    else:
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}, "
              f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
              f"time = {epoch_time:.2f} sec")

    if best_val_loss - val_loss > MIN_DELTA:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "../../models/transformer.pt")
        print(f"New best model saved at epoch {epoch + 1}, train_loss = {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"Эпох без улучшений: {epochs_no_improve}")

print(generate_title(model, tokenizer, "Статья о применении трансформеров для NLP"))
