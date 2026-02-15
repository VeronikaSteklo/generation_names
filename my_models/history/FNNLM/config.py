import torch

CONTEXT_SIZE = 8
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 30
MIN_WORD_FREQ = 10

TRAIN_DATASET = "../../../data/training_data/train_df.csv"
VAL_DATASET = "../../../data/training_data/val_df.csv"

LIMIT = 100000

PATIENCE = 3
COUNTER = 0
BEST_VAL_LOSS = float('inf')
save_path = "../../../models/fnnlm/best_fnnlm_model.pth"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Используемое устройство: {DEVICE}")
