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

METEOR: by default this uses nltk's meteor_score, a reimplementation that
is known to diverge (usually scores noticeably *higher*) from the official
METEOR 1.5 Java tool the paper actually used. Pass --meteor-jar to shell
out to the real jar instead for paper-comparable numbers -- see
tools/meteor-1.5/ (English-only jar + paraphrase table, same file used by
the MS-COCO caption-eval toolkit).

Usage:
    python scripts/evaluate.py --generations results/grnn_all_test.csv
    python scripts/evaluate.py --generations results/grnn_all_test.csv \
        --meteor-jar tools/meteor-1.5/meteor-1.5.jar
"""
import argparse
import subprocess
import sys
import tempfile
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


def official_meteor_sum(df, jar_path):
    """Corpus-level METEOR via the official 1.5 jar, returning (weighted_sum,
    n) rather than an average so callers can combine multiple groups (e.g.
    per-source -> overall) without re-invoking the jar. -r (reference count)
    must be fixed per invocation, but not every image has exactly 5
    references after cleaning, so we group by reference count within df and
    size-weight across those buckets -- an approximation of a single true
    corpus-level run, but the 5-ref rows are ~97% of the data. Each jar
    invocation costs ~15s (JVM start + loading the paraphrase table), so
    callers should call this once per group rather than per group per
    ref-count from the outside."""
    total_score, total_n = 0.0, 0
    for ref_count, sub in df.groupby(df["references"].apply(
            lambda s: len([r for r in str(s).split(" | ") if r.strip()]))):
        if ref_count == 0:
            continue
        hyp_lines, ref_lines = [], []
        for row in sub.itertuples():
            hyp_tokens = tokenize(row.generated)
            refs = [r for r in str(row.references).split(" | ") if r.strip()][:ref_count]
            if not hyp_tokens or len(refs) != ref_count:
                continue
            hyp_lines.append(" ".join(hyp_tokens))
            for r in refs:
                ref_lines.append(" ".join(tokenize(r)))
        if not hyp_lines:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            hyp_path = Path(tmp) / "hyp.txt"
            ref_path = Path(tmp) / "ref.txt"
            hyp_path.write_text("\n".join(hyp_lines) + "\n")
            ref_path.write_text("\n".join(ref_lines) + "\n")
            out = subprocess.run(
                ["java", "-Xmx2G", "-jar", str(jar_path), str(hyp_path), str(ref_path),
                 "-l", "en", "-norm", "-r", str(ref_count), "-q"],
                capture_output=True, text=True, check=True,
            )
        final_line = out.stdout.strip().splitlines()[-1]
        total_score += float(final_line) * len(hyp_lines)
        total_n += len(hyp_lines)
    return total_score, total_n


def score_group_bleu(df):
    smoothing = SmoothingFunction().method1
    hyps, refs_list = [], []
    for row in df.itertuples():
        hyp_tokens = tokenize(row.generated)
        ref_tokens = [tokenize(r) for r in str(row.references).split(" | ") if r.strip()]
        if not ref_tokens or not hyp_tokens:
            continue
        hyps.append(hyp_tokens)
        refs_list.append(ref_tokens)
    bleu = corpus_bleu(refs_list, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    return bleu, len(hyps)


def score_group_nltk_meteor(df):
    scores = []
    for row in df.itertuples():
        hyp_tokens = tokenize(row.generated)
        ref_tokens = [tokenize(r) for r in str(row.references).split(" | ") if r.strip()]
        if not ref_tokens or not hyp_tokens:
            continue
        scores.append(meteor_score(ref_tokens, hyp_tokens))
    return sum(scores), len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True, help="CSV produced by decode.py")
    ap.add_argument("--meteor-jar", default=None,
                     help="Path to meteor-1.5.jar -- use the official tool instead of nltk's "
                          "reimplementation for paper-comparable METEOR numbers")
    args = ap.parse_args()

    ensure_nltk_data()
    df = pd.read_csv(args.generations)
    meteor_fn = (lambda g: official_meteor_sum(g, args.meteor_jar)) if args.meteor_jar \
        else score_group_nltk_meteor

    print(f"{'source':<10}{'n':>6}{'BLEU':>10}{'METEOR':>10}")
    meteor_sum_all, meteor_n_all = 0.0, 0
    for source, group in df.groupby("source"):
        bleu, n = score_group_bleu(group)
        m_sum, m_n = meteor_fn(group)
        meteor_sum_all += m_sum
        meteor_n_all += m_n
        meteor = m_sum / max(m_n, 1)
        print(f"{source:<10}{n:>6}{bleu * 100:>10.1f}{meteor * 100:>10.1f}")
    overall_bleu, overall_n = score_group_bleu(df)
    overall_meteor = meteor_sum_all / max(meteor_n_all, 1)
    print(f"{'all':<10}{overall_n:>6}{overall_bleu * 100:>10.1f}{overall_meteor * 100:>10.1f}")


if __name__ == "__main__":
    main()
