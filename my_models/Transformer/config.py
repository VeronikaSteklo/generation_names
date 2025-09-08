import torch

# ----------------- Устройство -----------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Используемое устройство: {device}")

# ----------------- Параметры обучения -----------------
NUM_EPOCHS = 25
PATIENCE = 5
MIN_DELTA = 1e-3
BATCH_SIZE = 32
LR = 5e-4
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING=0.1

WARMUP_EPOCHS = 1
MAX_GRAD_NORM = 1.0
T_0 = 4
T_MULT = 2,
ETA_MIN = 1e-6

LIMIT=100

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