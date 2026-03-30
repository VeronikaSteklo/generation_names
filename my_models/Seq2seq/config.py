from __future__ import annotations

import torch

TRAIN_DATA_PATH = "../../data/training_data/train_df.csv"
VAL_DATA_PATH = "../../data/training_data/val_df.csv"
MODEL_SAVE_PATH = "../../models/seq2seq/best_model_seq2seq.pt"
VOCAB_PATH = "../../models/seq2seq/vocab.pkl"

MAX_VOCAB_SIZE = 50_000
MIN_COUNT = 10
LIMIT = 100_000
N_SAMPLES = 20

MAX_SRC_LEN = 300
MAX_TRG_LEN = 30
BATCH_SIZE = 32
NUM_EPOCHS = 10
CLIP = 1.0

ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 512
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.4
LR = 3e-4
PATIENCE = 2

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Используемое устройство: {device}")
