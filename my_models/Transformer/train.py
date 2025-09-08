import time

from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from my_models.Transformer.data.data_loader import TextTitleDataset, collate_fn
from my_models.Transformer.model.model import make_model, run_epoch, generate_title, evaluate_bleu

from config import *

model, tokenizer = make_model(N=MODEL_N, d_model=D_MODEL, d_ff=D_FF, h=NUM_HEADS, dropout=DROPOUT, device=device)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=T_0,
    T_mult=T_MULT,
    eta_min=ETA_MIN
)

best_val_loss = float("inf")
best_train_loss = float("inf")
epochs_no_improve = 0

train_dataset = TextTitleDataset(
    data_path=TRAIN_CSV,
    tokenizer=tokenizer,
    limit=LIMIT
)

val_dataset = TextTitleDataset(
    data_path=VAL_CSV,
    tokenizer=tokenizer,
    limit=LIMIT
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

num_batches = len(train_loader)

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train_loss = run_epoch(
        model=model,
        tokenizer=tokenizer,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        label_smoothing=LABEL_SMOOTHING,
        scheduler=scheduler,
        epoch=epoch,
        num_batches=num_batches,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_epochs=WARMUP_EPOCHS,
        base_lr=LR
    )

    val_loss = run_epoch(
        model=model,
        tokenizer=tokenizer,
        data_loader=val_loader,
        device=device,
        train=False
    )

    current_lr = optimizer.param_groups[0]["lr"]

    if (epoch + 1) % 5 == 0:
        bleu, bleu1, bleu4 = evaluate_bleu(
            model=model, tokenizer=tokenizer,
            data_loader=val_loader, device=device
        )
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch + 1}, "
              f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
              f"BLEU = {bleu:.4f}, BLEU1 = {bleu1:.4f}, BLEU4 = {bleu4:.4f}, "
              f"LR = {current_lr:.2e}, time = {epoch_time:.2f} sec")
    else:
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}, "
              f"train_loss = {train_loss:.4f}, val_loss = {val_loss:.4f}, "
              f"LR = {current_lr:.2e}, time = {epoch_time:.2f} sec")

    if best_val_loss - val_loss > MIN_DELTA:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': val_loss,
            'tokenizer': tokenizer,
            'epochs_no_improve': epochs_no_improve,
        }, MODEL_PATH)
        print(f"Новая модель с лучшими результатами сохранена на {epoch + 1} эпохе, val_loss = {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        print(f"Эпох без улучшений: {epochs_no_improve}")
        if epochs_no_improve > PATIENCE:
            print(f"Сработал ранний останов. Пройдено {epoch + 1} эпох. Лучший результат: {best_val_loss:.4f}")
            break

print(generate_title(model, tokenizer, "Статья о применении трансформеров для NLP"))
