import torch

# ----------------- Устройство -----------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Используемое устройство: {device}")

# ----------------- Параметры обучения -----------------
NUM_EPOCHS = 200
PATIENCE = 5
MIN_DELTA = 1e-3
BATCH_SIZE = 32
LR = 5e-4
DROPOUT = 0.2
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING=0.1

WARMUP_EPOCHS = 1
MAX_GRAD_NORM = 1.0
T_0 = 10
T_MULT = 2
ETA_MIN = 1e-6

LIMIT=100

# ----------------- Пути-----------------
# MODEL_PATH = "../../models/transformer_all.pth"
# TRAIN_CSV = "../../data/clean_data_aug/train_df.csv"
# VAL_CSV = "../../data/clean_data_aug/val_df.csv"

MODEL_PATH = "../../models/debug.pth"
TRAIN_CSV = "debug/debug.csv"
VAL_CSV = "debug/debug.csv"

# ----------------- Модель -----------------
MODEL_N = 1
D_MODEL = 64
D_FF = 128
NUM_HEADS = 2

# ----------------- Padding -----------------
PAD_TOKEN = "<pad>"