import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial

import config
from data.dataset import TitleDataset, collate_fn
from model.model import TitleRNN
from model.utils import generate_title

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class ModelEvaluator:
    def __init__(self):
        self.rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1

    def calculate_perplexity(self, losses):
        return np.exp(np.mean(losses))

    def calculate_bleu(self, references, hypotheses):
        return corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothie)

    def calculate_rouge(self, references, hypotheses):
        total_rouge_l = 0
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge_evaluator.score(ref, hyp)
            total_rouge_l += scores['rougeL'].fmeasure
        return total_rouge_l / len(references) if len(references) > 0 else 0

    def calculate_meteor(self, references, hypotheses):
        total_meteor = 0
        for ref, hyp in zip(references, hypotheses):
            total_meteor += meteor_score(ref, hyp)
        return total_meteor / len(references) if len(references) > 0 else 0


def save_results_to_csv(metrics, model_name, output_path):
    row = {
        "model_name": model_name,
        **metrics
    }
    df_new = pd.DataFrame([row])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)
    if file_exists:
        with open(output_path, 'a', encoding='utf-8-sig') as f:
            f.write('\n')

    df_new.to_csv(output_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    print(f"\nРезультаты сохранены в: {output_path}")


def run_rnn_evaluation(limit=500):
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
    vocab = checkpoint['vocab']

    model = TitleRNN(
        vocab_size=len(vocab.itos),
        emb_dim=config.EMBED_DIM,
        hid_dim=config.HID_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_df = pd.read_csv(config.VAL_DATA_PATH).head(limit)
    val_dataset = TitleDataset(val_df, vocab)

    pad_idx = vocab.stoi[config.PAD_TOKEN]
    collate_p = partial(collate_fn, pad_idx=pad_idx)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_p)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    evaluator = ModelEvaluator()

    losses = []
    all_refs_tokens = []
    all_hyps_tokens = []
    all_refs_text = []
    all_hyps_text = []

    with torch.no_grad():
        for text_tensor, title_tensor in tqdm(val_loader, desc="Evaluation"):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            text_tensor, title_tensor = text_tensor.to(config.DEVICE), title_tensor.to(config.DEVICE)

            output = model(text_tensor, title_tensor[:, :-1])
            output_dim = output.shape[-1]
            loss = criterion(output.reshape(-1, output_dim), title_tensor[:, 1:].reshape(-1))
            losses.append(loss.item())

    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Generating"):
        raw_text = str(row['text'])
        raw_title = str(row['title'])

        gen_title = generate_title(model, raw_text, vocab, config.DEVICE)

        ref_tokens = vocab.tokenize(raw_title)
        hyp_tokens = vocab.tokenize(gen_title)

        all_refs_tokens.append([ref_tokens])
        all_hyps_tokens.append(hyp_tokens)
        all_refs_text.append(raw_title)
        all_hyps_text.append(gen_title)

    metrics = {
        "PPL": evaluator.calculate_perplexity(losses),
        "BLEU-4": evaluator.calculate_bleu(all_refs_tokens, all_hyps_tokens),
        "ROUGE-L": evaluator.calculate_rouge(all_refs_text, all_hyps_text),
        "METEOR": evaluator.calculate_meteor(all_refs_tokens, all_hyps_tokens)
    }

    return metrics


if __name__ == "__main__":
    MODEL_NAME = "RNN"
    OUTPUT_CSV = "../../../outputs/results.csv"

    results = run_rnn_evaluation(limit=500)

    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")

    save_results_to_csv(results, MODEL_NAME, OUTPUT_CSV)
