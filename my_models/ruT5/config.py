import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, T5Tokenizer

from my_models.ruT5.data.dataset import T5TitleDataset

# ==========================================
# Гиперпараметры (Config)
# ==========================================

EPOCHS = 3
BATCH_SIZE = 8
LR = 5e-5
WARMUP_STEPS = 500
MAX_SOURCE_TEXT_LEN = 512
MAX_TARGET_TITLE_LEN = 32

PATIENCE = 2
BEST_VAL_LOSS = float('inf')

MODEL_NAME = "cointegrated/rut5-small"
TOKENIZER = T5Tokenizer.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Используемый девайс: {DEVICE}")

TRAIN_PATH = "../../data/training_data/train_df.csv"
VAL_PATH = "../../data/training_data/train_df.csv"
MODEL_PATH = "../../models/ruT5-small/"

train_dataset = T5TitleDataset(
    df=pd.read_csv(TRAIN_PATH).sample(30000, random_state=42),
    tokenizer=TOKENIZER,
    max_src_len=MAX_SOURCE_TEXT_LEN,
    max_tgt_len=MAX_TARGET_TITLE_LEN
)
validation = pd.read_csv(VAL_PATH)
validation.text = validation.text.apply(lambda x: ".".join(x.split(".", 3)[:3]).strip() + ".")
val_dataset = T5TitleDataset(
    df=validation.sample(3000, random_state=42),
    tokenizer=TOKENIZER,
    max_src_len=MAX_SOURCE_TEXT_LEN,
    max_tgt_len=MAX_TARGET_TITLE_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# Функции инициализации
# ==========================================

def get_optimizer_and_scheduler(model, train_loader_len, epochs=EPOCHS, lr=LR, warmup_steps=WARMUP_STEPS):
    """
    Функция для инициализации оптимизатора и scheduler.
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = train_loader_len * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    return optimizer, scheduler
