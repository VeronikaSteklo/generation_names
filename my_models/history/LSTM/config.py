import torch

TRAIN_DATA_PATH = "../../../data/training_data/train_df.csv"
VAL_DATA_PATH = "../../../data/training_data/val_df.csv"
MODEL_SAVE_PATH = "../../../models/lstm/lstm_model.pt"

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 20
PATIENCE = 3
MAX_LEN = 150

LIMIT = 100000

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")