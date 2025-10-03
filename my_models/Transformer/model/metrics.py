import torch
import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
nltk.download('wordnet')

from my_models.Transformer.model.utils import generate_title


def evaluate_bleu(model, tokenizer, data_loader, device, mode="greedy"):
    model.eval()
    refs = []
    hyps = []

    with torch.no_grad():
        for batch in data_loader:
            src_texts = batch["src_text"]
            tgt_titles = batch["tgt_text"]

            for text, ref_title in zip(src_texts, tgt_titles):
                gen_title = generate_title(model, tokenizer, text, device=device, mode=mode)

                refs.append([ref_title.split()])
                hyps.append(gen_title.split())

    smooth_fn = SmoothingFunction().method1
    bleu = corpus_bleu(refs, hyps, smoothing_function=smooth_fn)
    bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smooth_fn)
    return bleu, bleu1

def evaluate_meteor(model, tokenizer, data_loader, device, mode="greedy"):
    model.eval()
    refs = []
    hyps = []

    with torch.no_grad():
        for batch in data_loader:
            src_texts = batch["src_text"]
            tgt_titles = batch["tgt_text"]

            for text, ref_title in zip(src_texts, tgt_titles):
                gen_title = generate_title(model, tokenizer, text, device=device, mode=mode)

                refs.append([ref_title.split()])
                hyps.append(gen_title.split())

    scores = []
    for refs, hyps in zip(refs, hyps):
        score = meteor_score(refs, hyps)
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0