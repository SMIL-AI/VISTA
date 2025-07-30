import nltk
import numpy as np
from nltk import bleu_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.tokenize import TreebankWordTokenizer

from cider.cider import Cider


try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpus/wordnet')
except:
    nltk.download('wordnet', quiet=True)


def tokenize_sentence(sentence):
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(sentence)
    if len(words) == 0:
        return ""
    return " ".join(words)


# Compute BLEU-4 score on a single sentence
def compute_bleu_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return sentence_bleu([tokenized_reference], tokenized_hypothesis,
                         weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=bleu_score.SmoothingFunction().method3)


# Compute METEOR score on a single sentence
def compute_meteor_single(tokenized_hypothesis, tokenized_reference):
    # convert tokenized sentence (joined by spaces) into list of words
    tokenized_hypothesis = tokenized_hypothesis.split(" ")
    tokenized_reference = tokenized_reference.split(" ")

    return meteor_score([tokenized_reference], tokenized_hypothesis)


# Compute ROUGE-L score on a single sentence
def compute_rouge_l_single(sentence_hypothesis, sentence_reference):
    rouge_l_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = rouge_l_scorer.score(sentence_hypothesis, sentence_reference)
    rouge_l_score = rouge_score['rougeL']
    return rouge_l_score.fmeasure


# Compute CIDEr score on a single sentence
def compute_cider_single(sentence_hypothesis, sentence_reference):
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score([sentence_reference], [sentence_hypothesis])

    return cider_score


# Compute metrics based for a single caption.
def compute_metrics_single(pred, gt):
    tokenized_pred = tokenize_sentence(pred)
    tokenized_gt = tokenize_sentence(gt)

    bleu_score = compute_bleu_single(tokenized_pred, tokenized_gt)
    meteor_score = compute_meteor_single(tokenized_pred, tokenized_gt)
    rouge_l_score = compute_rouge_l_single(pred, gt)
    cider_score = compute_cider_single([tokenized_pred], [tokenized_gt])

    return {
        "bleu": bleu_score,
        "meteor": meteor_score,
        "rouge-l": rouge_l_score,
        "cider": cider_score,
    }


def compute_avg(metrics_list):
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0].keys()} if metrics_list else {}
# Convert a list containing annotation of all segments of a scenario to a dict keyed by segment label.
#   - Example input (segment_list):
#         [
#             {
#                 "labels": [
#                     "0"
#                 ],
#                 "caption_pedestrian": "",
#                 "caption_vehicle": ""
#             },
#             {
#                 ...
#             }
#         ]
#   - Example output (segment_dict):
#         {
#             "0": {
#                 "caption_pedestrian": "",
#                 "caption_vehicle": ""
#             },
#             ...
#         }
def convert_to_dict(segment_list):
    segment_dict = {}
    for segment in segment_list:
        segment_number = segment["labels"][0]

        segment_dict[segment_number] = {
            "caption_pedestrian": segment["caption_pedestrian"],
            "caption_vehicle": segment["caption_vehicle"]
        }

    return segment_dict


if __name__ == "__main__":
    # Example usage
    pred = "The quick brown fox jumps over the lazy dog."
    gt = "A fast brown fox leaps over a lazy dog."

    metrics = compute_metrics_single(pred, gt)
    print(metrics)

    # Example segment list to dict conversion
    segment_list = [
        {
            "labels": ["0"],
            "caption_pedestrian": "Pedestrian walking",
            "caption_vehicle": "Car passing by"
        },
        {
            "labels": ["1"],
            "caption_pedestrian": "Bicycle rider",
            "caption_vehicle": "Truck stopping"
        }
    ]

    segment_dict = convert_to_dict(segment_list)
    print(segment_dict)