"""
evaluate.py
---------------------------------
Computes BLEU (4-gram, equal weights) and METEOR of generated questions
against the 5 human reference questions per image, following the paper's
own evaluation choice (Section 5.2): BLEU is used as the primary metric
since it correlates with human judgment nearly as well as the paper's
∆BLEU, which needs extra per-reference human ratings we don't have (see
README.md "Known deviations").

Reports per-source (bing/coco/flickr) BLEU + METEOR in the same
row=source layout as the paper's Table 5, for a (not exactly apples-to-
apples, given our smaller dataset) point of comparison against it.

Usage:
    python scripts/evaluate.py --generations results/grnn_all_test.csv
"""
import argparse
import sys
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vqg.vocab import tokenize


def ensure_nltk_data():
    for pkg in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


def score_group(df):
    smoothing = SmoothingFunction().method1
    hyps, refs_list, meteor_scores = [], [], []
    for row in df.itertuples():
        hyp_tokens = tokenize(row.generated)
        ref_tokens = [tokenize(r) for r in str(row.references).split(" | ") if r.strip()]
        if not ref_tokens or not hyp_tokens:
            continue
        hyps.append(hyp_tokens)
        refs_list.append(ref_tokens)
        meteor_scores.append(meteor_score(ref_tokens, hyp_tokens))

    bleu = corpus_bleu(refs_list, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    meteor = sum(meteor_scores) / max(len(meteor_scores), 1)
    return bleu, meteor, len(hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="CSV produced by decode.py")
    args = ap.parse_args()

    ensure_nltk_data()
    df = pd.read_csv(args.generations)

    print(f"{'source':<10}{'n':>6}{'BLEU':>10}{'METEOR':>10}")
    for source, group in df.groupby("source"):
        bleu, meteor, n = score_group(group)
        print(f"{source:<10}{n:>6}{bleu * 100:>10.1f}{meteor * 100:>10.1f}")
    overall_bleu, overall_meteor, overall_n = score_group(df)
    print(f"{'all':<10}{overall_n:>6}{overall_bleu * 100:>10.1f}{overall_meteor * 100:>10.1f}")


if __name__ == "__main__":
    main()
