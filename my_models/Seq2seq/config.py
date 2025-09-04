from __future__ import annotations

DATA_CSV = "../data/data_tokenize.csv"
SRC_VOCAB_PATH = "../data/vocabs/src_vocab.pkl"
TRG_VOCAB_PATH = "../data/vocabs/trg_vocab.pkl"
BEST_MODEL_PATH = "../models/best_model_seq2seq.pt"

MAX_VOCAB_SIZE = 30_000
MAX_INPUT_LEN = 300
MAX_TARGET_LEN = 30

PAD_IDX = 0
HID_DIM = 512
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
ENC_DROPOUT = 0.4
DEC_DROPOUT = 0.4
N_LAYERS = 1

BATCH_SIZE = 32
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
CLIP = 1.0
EARLY_STOPPING_PATIENCE = 1
LABEL_SMOOTHING = 0.1

SEED = 42