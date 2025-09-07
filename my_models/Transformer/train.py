import time

from torch import optim
from torch.utils.data import DataLoader
from my_models.Transformer.data.data_loader import TextTitleDataset, collate_fn
from my_models.Transformer.model.model import make_model, run_epoch, generate_title, evaluate_bleu

from config import *

model, tokenizer = make_model(N=MODEL_N, d_model=D_MODEL, d_ff=D_FF, h=NUM_HEADS, dropout=DROPOUT, device=device)
optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

best_val_loss = float("inf")
best_train_loss = float("inf")
epochs_no_improve = 0

train_dataset = TextTitleDataset(
    data_path=TRAIN_CSV,
    tokenizer=tokenizer,
    limit=100
)

val_dataset = TextTitleDataset(
    data_path=VAL_CSV,
    tokenizer=tokenizer,
    limit=100
)

pad_id = tokenizer.token_2_idx(PAD_TOKEN)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, pad_id, device=device)
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=lambda batch: collate_fn(batch, pad_id, device=device)
)
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = run_epoch(
        model=model, tokenizer=tokenizer, data_loader=train_loader,
        optimizer=optimizer, device=device
    )

    val_loss = run_epoch(
        model=model, tokenizer=tokenizer,
        data_loader=val_loader, optimizer=None,
        device=device, train=False
    )

    if (epoch + 1) % 5 == 0:
        bleu, bleu1, bleu4 = evaluate_bleu(
            model=model, tokenizer=tokenizer,
            data_loader=val_loader, device=device
        )
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'tokenizer': tokenizer,
        }, MODEL_PATH)
        print(f"New best model saved at epoch {epoch + 1}, train_loss = {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"Эпох без улучшений: {epochs_no_improve}")
        if epochs_no_improve > PATIENCE:
            break

print(generate_title(model, tokenizer, "Статья о применении трансформеров для NLP"))
