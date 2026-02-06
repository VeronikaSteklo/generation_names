from transformers import T5ForConditionalGeneration
from config import *
from model.train import train_epoch, validate_epoch

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)

patience_counter = 0
prev_val_loss = BEST_VAL_LOSS

optimizer, scheduler = get_optimizer_and_scheduler(model, len(train_loader))

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
    val_loss = validate_epoch(model, val_loader, DEVICE, TOKENIZER)
    improvement = prev_val_loss - val_loss if prev_val_loss != float('inf') else 0
    print(
        f"\nEpoch: {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Улучшение: {improvement:.4f}"
    )
    prev_val_loss = val_loss

    if val_loss < BEST_VAL_LOSS:
        BEST_VAL_LOSS = val_loss
        patience_counter = 0
        model.save_pretrained(MODEL_PATH)
        TOKENIZER.save_pretrained(MODEL_PATH)
    else:
        patience_counter += 1
        print(f"Прогресса нет. Терпение: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Ранняя остановка: лосс замер. Прекращаем обучение.")
        break
