from __future__ import annotations

import torch

TRAIN_DATA_PATH = "../../data/training_data/train_df.csv"
VAL_DATA_PATH = "../../data/training_data/val_df_chunked.csv"
MODEL_SAVE_PATH = "../../models/seq2seq/best_model_seq2seq.pt"
VOCAB_PATH = "../../models/seq2seq/vocab.pkl"

MAX_VOCAB_SIZE = 100_000
MIN_COUNT = 1
LIMIT = 200_000
N_SAMPLES = 10

MAX_SRC_LEN = 300
MAX_TRG_LEN = 30
BATCH_SIZE = 128
NUM_EPOCHS = 30
CLIP = 1.0

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 256
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
LR = 5e-4
PATIENCE = 4

SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Используемое устройство: {device}")
