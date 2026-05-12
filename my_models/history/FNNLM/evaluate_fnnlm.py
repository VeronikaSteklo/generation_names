import torch
import pandas as pd
import numpy as np
import os
import pickle
import re
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from data.dataset import FNNLMDataset, first_n_sentences
from model.model import FNNLM
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


def run_fnnlm_evaluation(limit_rows=200):
    dataset_info_path = "../../../models/fnnlm/dataset_info.pkl"
    with open(dataset_info_path, 'rb') as f:
        ds_info = pickle.load(f)

    vocab = ds_info['vocab']
    rev_vocab = ds_info['rev_vocab']
    vocab_size = ds_info['vocab_size']

    model = FNNLM(vocab_size, config.EMBEDDING_DIM, config.HIDDEN_DIM).to(config.DEVICE)
    model.load_state_dict(torch.load(config.save_path, map_location=config.DEVICE))
    model.eval()

    val_dataset = FNNLMDataset(config.VAL_DATASET, config.CONTEXT_SIZE, vocab=vocab, first_sentences=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    evaluator = ModelEvaluator()

    losses = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="PPL Calculation"):
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())

    df_val = pd.read_csv(config.VAL_DATASET).head(limit_rows)

    all_refs_tokens = []
    all_hyps_tokens = []
    all_refs_text = []
    all_hyps_text = []

    for _, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Text Metrics"):
        raw_text = first_n_sentences(row['text'], n=2)
        raw_title = str(row['title'])

        gen_title = generate_title(model, raw_text, vocab, rev_vocab)

        ref_tokens = re.sub(r'[^\w\s]', '', raw_title.lower()).split()
        hyp_tokens = gen_title.split()

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
    MODEL_NAME = "FNNLM"
    OUTPUT_CSV = "../../../outputs/results.csv"

    results = run_fnnlm_evaluation(limit_rows=300)

    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")

    save_results_to_csv(results, MODEL_NAME, OUTPUT_CSV)
