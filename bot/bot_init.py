import torch
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from my_models.seq2seq import vocab

from bot.config import config

bot = Bot(
    token=config.bot_token.get_secret_value(),
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_seq2esq = torch.load(
    "/Users/veronika_steklo/PycharmProjects/generation_names/models/best_model_seq2seq.pt",
    map_location=device
)

src_vocab = vocab.Vocab.load("/Users/veronika_steklo/PycharmProjects/generation_names/data/vocabs/src_vocab.pkl")
trg_vocab = vocab.Vocab.load("/Users/veronika_steklo/PycharmProjects/generation_names/data/vocabs/trg_vocab.pkl")