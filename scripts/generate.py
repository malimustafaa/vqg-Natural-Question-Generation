"""
generate.py
---------------------------------
Ask a question about a single image with a trained GRNN checkpoint -- the
"try it yourself" entry point, as opposed to decode.py (which scores an
entire manifest against precomputed features and human references). Runs
the exact same VGG16 fc7 extraction as extract_features.py, computed fresh
from the raw image file -- no precomputed .pt needed.

Usage:
    python scripts/generate.py --image path/to/photo.jpg \
        --checkpoint checkpoints/grnn_all/best.pt --vocab data/raw/vocab.json
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_features import VGG16Fc7, load_image  # noqa: E402
from vqg.beam_search import beam_search_decode  # noqa: E402
from vqg.model import GRNN  # noqa: E402
from vqg.vocab import Vocab  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vocab", default="data/raw/vocab.json")
    ap.add_argument("--beam-width", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--length-penalty", type=float, default=0.6,
                     help="Matches the value used for results/*_test.csv; 0.0 = paper's raw-log-prob "
                          "beam search (see README).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    vocab = Vocab.load(args.vocab)

    cnn = VGG16Fc7().to(args.device)
    image_tensor = load_image(args.image).unsqueeze(0).to(args.device)
    feat = cnn(image_tensor).squeeze(0)

    model = GRNN(vocab_size=len(vocab), pad_id=vocab.pad_id).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()

    question, score = beam_search_decode(
        model, feat, vocab, beam_width=args.beam_width, max_len=args.max_len,
        length_penalty=args.length_penalty,
    )
    print(question)


if __name__ == "__main__":
    main()
