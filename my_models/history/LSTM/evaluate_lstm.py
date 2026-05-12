import torch
import pandas as pd
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from data.dataset import LSTMTitleDataset
from model.model import LSTMTitleGen
from model.utils import generate_title

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class ModelEvaluator:
    def __init__(self):
        self.rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1

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
            if os.path.getsize(output_path) > 0:
                f.write('\n')

    df_new.to_csv(output_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')
    print(f"\nРезультаты сохранены в: {output_path}")


def run_evaluation(limit_rows=300):
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
    vocab = checkpoint['vocab']

    model = LSTMTitleGen(
        vocab_size=len(vocab),
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    val_full = pd.read_csv(config.VAL_DATA_PATH).dropna(subset=['text', 'title'])
    val_df_sample = val_full.sample(n=min(limit_rows * 2, len(val_full)), random_state=42)
    val_dataset = LSTMTitleDataset(val_df_sample, vocab, config.MAX_LEN)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    evaluator = ModelEvaluator()

    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets, masks in tqdm(val_loader, desc="PPL"):
            inputs, targets, masks = inputs.to(config.DEVICE), targets.to(config.DEVICE), masks.to(config.DEVICE)

            logits, _ = model(inputs)

            raw_loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            masked_loss = raw_loss * masks.view(-1)

            total_loss += masked_loss.sum().item()
            total_tokens += masks.sum().item()

    avg_loss = total_loss / (total_tokens + 1e-9)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    all_refs_tokens = []
    all_hyps_tokens = []
    all_refs_text = []
    all_hyps_text = []

    gen_samples = val_full.head(limit_rows)

    for _, row in tqdm(gen_samples.iterrows(), total=len(gen_samples), desc="Metrics"):
        raw_text = str(row['text'])
        raw_title = str(row['title'])

        gen_title = generate_title(model, vocab, config.DEVICE, raw_text)

        ref_tokens = raw_title.lower().split()
        hyp_tokens = gen_title.lower().split()

        all_refs_tokens.append([ref_tokens])
        all_hyps_tokens.append(hyp_tokens)
        all_refs_text.append(raw_title)
        all_hyps_text.append(gen_title)

    metrics = {
        "PPL": ppl,
        "BLEU-4": evaluator.calculate_bleu(all_refs_tokens, all_hyps_tokens),
        "ROUGE-L": evaluator.calculate_rouge(all_refs_text, all_hyps_text),
        "METEOR": evaluator.calculate_meteor(all_refs_tokens, all_hyps_tokens)
    }

    return metrics


if __name__ == "__main__":
    MODEL_NAME = "LSTM_Bidirectional"
    OUTPUT_CSV = "../../../outputs/results.csv"

    results = run_evaluation(limit_rows=90)

    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")

    save_results_to_csv(results, MODEL_NAME, OUTPUT_CSV)
