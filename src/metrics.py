"""
Lexical evaluation metrics: ROUGE-1 and sentence-level BLEU.

These complement BERTScore, matching the metrics reported in the
MultiNativQA paper (BERTScore F1, BLEU, ROUGE-1).

Answers are in English, Hindi and Bengali, so tokenization is
Unicode-aware: punctuation (including the Devanagari danda) is
replaced by spaces and text is split on whitespace. Python's \\w is
NOT used because it breaks Indic words at combining vowel signs.
Scores are in the 0-1 range.
"""

import unicodedata
from collections import Counter
from typing import List


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation/symbols, split on whitespace."""

    cleaned = "".join(
        " " if unicodedata.category(ch)[0] in ("P", "S") else ch
        for ch in text.lower()
    )

    return cleaned.split()


def rouge1_f1(prediction: str, reference: str) -> float:
    """Unigram-overlap F1 between a prediction and a reference."""

    pred_counts = Counter(tokenize(prediction))
    ref_counts = Counter(tokenize(reference))

    if not pred_counts or not ref_counts:
        return 0.0

    overlap = sum((pred_counts & ref_counts).values())

    if overlap == 0:
        return 0.0

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(ref_counts.values())

    return 2 * precision * recall / (precision + recall)


def sentence_bleu(prediction: str, reference: str) -> float:
    """
    Sentence-level BLEU (sacreBLEU, exponential smoothing) on the
    Unicode-aware tokens above, scaled to 0-1.
    """

    try:
        from sacrebleu.metrics import BLEU
    except ImportError as e:
        raise ImportError(
            "sacrebleu is not installed. "
            "Install with: pip install sacrebleu"
        ) from e

    pred_tokens = " ".join(tokenize(prediction))
    ref_tokens = " ".join(tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    # tokenize="none": tokens are already split on whitespace
    bleu = BLEU(tokenize="none", effective_order=True)

    return bleu.sentence_score(pred_tokens, [ref_tokens]).score / 100.0
