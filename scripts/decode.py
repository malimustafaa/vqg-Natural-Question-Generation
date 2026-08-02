"""
decode.py
---------------------------------
Runs beam search (width 8, <unk> banned) over a manifest+features file using
a trained GRNN checkpoint, and writes generated questions to CSV alongside
the human reference questions, for evaluate.py to score.

Usage:
    python scripts/decode.py --checkpoint checkpoints/grnn_all/best.pt \
        --manifest data/raw/manifest_test.csv --features data/features/test.pt \
        --vocab data/raw/vocab.json --out results/grnn_all_test.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vqg.beam_search import beam_search_decode
from vqg.model import GRNN
from vqg.vocab import Vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--vocab", default="data/raw/vocab.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--length-penalty", type=float, default=0.0,
                     help="GNMT-style length normalization alpha (0.0 = off, matches the paper's "
                          "unspecified-either-way raw-log-prob beam search; ~0.6-1.0 counteracts "
                          "the brevity bias that favors short generic completions)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)
    features = torch.load(args.features)
    df = pd.read_csv(args.manifest)

    model = GRNN(vocab_size=len(vocab), pad_id=vocab.pad_id).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    rows = []
    for row in tqdm(df.itertuples(), total=len(df), desc="decoding"):
        if row.image_id not in features:
            continue
        feat = features[row.image_id].to(args.device)
        question, score = beam_search_decode(
            model, feat, vocab, beam_width=args.beam_width, max_len=args.max_len,
            length_penalty=args.length_penalty,
        )
        rows.append({
            "image_id": row.image_id,
            "source": row.source,
            "generated": question,
            "references": row.questions,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Decoded {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
