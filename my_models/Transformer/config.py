import torch

# ----------------- Устройство -----------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Используемое устройство: {device}")

# ----------------- Параметры обучения -----------------
NUM_EPOCHS = 25
PATIENCE = 2
MIN_DELTA = 2e-2
BATCH_SIZE = 32
LR = 1e-3
DROPOUT = 0.2
WEIGHT_DECAY = 1e-5

# ----------------- Пути-----------------
MODEL_PATH = "../../models/transformer_all.pth"
TRAIN_CSV = "../../data/all_data_augmented/train_df.csv"
VAL_CSV = "../../data/all_data_augmented/val_df.csv"

# ----------------- Модель -----------------
MODEL_N = 2
D_MODEL = 128
D_FF = 256
NUM_HEADS = 4

# ----------------- Padding -----------------
PAD_TOKEN = "<pad>"