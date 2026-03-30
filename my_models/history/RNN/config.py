import torch

TRAIN_DATA_PATH = "../../../data/training_data/train_df.csv"
VAL_DATA_PATH = "../../../data/training_data/val_df.csv"
MODEL_SAVE_PATH = "../../../models/rnn/rnn_model.pt"

LIMIT = 100_000

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

EMBED_DIM = 128
HID_DIM = 256
N_LAYERS = 2
DROPOUT = 0.3
LR = 1e-3
BATCH_SIZE = 16
EPOCHS = 15
PATIENT = 2

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"