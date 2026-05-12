import torch
import pandas as pd
import os
import math
from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from data.vocab import simple_tokenize
from data.dataset import TitleDataset, collate_fn
from model.utils import load_model, generate_response

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
    print(f"\nРезультаты добавлены в: {output_path}")


def run_seq2seq_evaluation(limit_rows=200):
    model, vocab = load_model(vocab_path=config.VOCAB_PATH, model=config.MODEL_PATH)
    model.to(config.device)

    val_df = pd.read_csv(config.VAL_DATA_PATH).dropna(subset=['text', 'title'])
    val_dataset = TitleDataset(val_df.head(limit_rows * 2), vocab, config.MAX_SRC_LEN, config.MAX_TRG_LEN)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for src, trg in tqdm(val_loader, desc="PPL"):
            src, trg = src.to(config.device), trg.to(config.device)

            outputs = model(src, trg, teacher_forcing_ratio=0.0)

            out_dim = outputs.shape[-1]

            outputs = outputs[:, 1:].reshape(-1, out_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg_flat)

            mask = (trg_flat != 0).float()
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / (total_tokens + 1e-9)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    evaluator = ModelEvaluator()
    all_refs_tokens = []
    all_hyps_tokens = []
    all_refs_text = []
    all_hyps_text = []

    test_samples = val_df.head(limit_rows)

    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples), desc="Generating"):
        raw_text = str(row['text'])
        raw_title = str(row['title'])

        gen_title = generate_response(model, vocab, raw_text, beam_width=3)

        ref_tokens = simple_tokenize(raw_title.lower())
        hyp_tokens = simple_tokenize(gen_title.lower())

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
    MODEL_NAME = "Seq2Seq_Attention_BeamSearch"
    OUTPUT_CSV = "../../outputs/results.csv"

    results = run_seq2seq_evaluation(limit_rows=200)

    for metric, val in results.items():
        print(f"{metric}: {val:.4f}")

    save_results_to_csv(results, MODEL_NAME, OUTPUT_CSV)
