"""
build_vocab.py
---------------------------------
Builds the VQG vocabulary from the training split's questions, following the
paper's rule exactly: keep every word appearing >=3 times in training
(Section 4.1). The paper's full corpus produced 1,942 tokens; ours will
differ since our training corpus is smaller due to 2016 link rot -- see
README.md "Known deviations".

Usage:
    python scripts/build_vocab.py --train-csv data/raw/all_train.csv --out data/raw/vocab.json
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vqg.vocab import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data/raw/all_train.csv")
    ap.add_argument("--out", default="data/raw/vocab.json")
    ap.add_argument("--min-freq", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.train_csv)
    all_questions = []
    for cell in df["questions"].dropna():
        all_questions.extend(str(cell).split(" | "))

    vocab = Vocab.build(all_questions, min_freq=args.min_freq)
    vocab.save(args.out)
    print(
        f"Built vocab: {len(vocab)} tokens (incl. 4 special) from {len(all_questions)} questions "
        f"across {len(df)} images -> {args.out}"
    )


if __name__ == "__main__":
    main()
