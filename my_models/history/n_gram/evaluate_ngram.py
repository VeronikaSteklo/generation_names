import os

import pandas as pd
import numpy as np
from tqdm import tqdm
from n_gram_model import TitleNgramModel, tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


class ModelEvaluator:
    def __init__(self):
        self.rouge_evaluator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1

    def calculate_perplexity(self, losses):
        """PPL = exp(average_loss)"""
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


def evaluate_ngram_performance(model, val_df):
    evaluator = ModelEvaluator()

    losses = []
    all_refs_tokens = []
    all_hyps_tokens = []
    all_refs_text = []
    all_hyps_text = []

    print("Начинаю генерацию и расчет метрик...")
    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        text = str(row['text'])
        true_title = str(row['title'])

        loss = model.calculate_sentence_loss(text, true_title)
        losses.append(loss)

        gen_title = model.generate_title(text)

        ref_tokens = tokenize(true_title)
        hyp_tokens = tokenize(gen_title)

        all_refs_tokens.append([ref_tokens])
        all_hyps_tokens.append(hyp_tokens)

        all_refs_text.append(true_title)
        all_hyps_text.append(gen_title)

    metrics = {
        "PPL": evaluator.calculate_perplexity(losses),
        "BLEU-4": evaluator.calculate_bleu(all_refs_tokens, all_hyps_tokens),
        "ROUGE-L": evaluator.calculate_rouge(all_refs_text, all_hyps_text),
        "METEOR": evaluator.calculate_meteor(all_refs_tokens, all_hyps_tokens)
    }
    return metrics


def save_results_to_csv(metrics, model_name, output_path):
    row = {
        "model_name": model_name,
        **metrics
    }

    df_new = pd.DataFrame([row])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_exists = os.path.isfile(output_path)

    df_new.to_csv(output_path, mode='a', index=False, header=not file_exists, encoding='utf-8-sig')


if __name__ == "__main__":
    MODEL_NAME = "TitleNgramModel_3gram"
    model_path = "../../../models/history/title_ngram_model.pkl"
    model = TitleNgramModel.load(model_path)

    val_df = pd.read_csv("../../../data/training_data/val_df.csv").head(100)
    OUTPUT_CSV = "../../../outputs/results.csv"

    results = evaluate_ngram_performance(model, val_df)

    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    save_results_to_csv(results, MODEL_NAME, OUTPUT_CSV)
