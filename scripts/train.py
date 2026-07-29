"""
train.py
---------------------------------
Trains the GRNN (vqg/model.py) with teacher forcing, SGD, and early stopping
on validation loss -- matching the paper's stated training recipe (Section
4.1). Learning rate / batch size / epoch count aren't specified in the paper,
so they're CLI flags here, meant to be tuned against validation BLEU.

Checkpointing is done every epoch (and whenever validation loss improves) so
that Colab session disconnects don't lose progress; --resume picks back up
from the last checkpoint.

Usage:
    python scripts/train.py --sources all \
        --train-manifest data/raw/manifest_train.csv --train-features data/features/train.pt \
        --val-manifest data/raw/manifest_val.csv --val-features data/features/val.pt \
        --vocab data/raw/vocab.json --checkpoint-dir checkpoints/grnn_all

    # GRNN_X (per-source), e.g. bing-only:
    python scripts/train.py --sources bing --checkpoint-dir checkpoints/grnn_bing ...
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vqg.dataset import VQGDataset, make_collate_fn
from vqg.model import GRNN
from vqg.vocab import Vocab


def run_epoch(model, loader, vocab, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="sum")

    for feats, inputs, targets in loader:
        feats, inputs, targets = feats.to(device), inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(feats, inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        n_tokens = (targets != vocab.pad_id).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["bing", "coco", "flickr"],
                     help="Which source(s) to train on -- omit/'all 3' for GRNN_all, one for GRNN_X")
    ap.add_argument("--train-manifest", default="data/raw/manifest_train.csv")
    ap.add_argument("--train-features", default="data/features/train.pt")
    ap.add_argument("--val-manifest", default="data/raw/manifest_val.csv")
    ap.add_argument("--val-features", default="data/features/val.pt")
    ap.add_argument("--vocab", default="data/raw/vocab.json")
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs w/o val improvement)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / "last.pt"
    best_ckpt = ckpt_dir / "best.pt"

    vocab = Vocab.load(args.vocab)
    train_features = torch.load(args.train_features)
    val_features = torch.load(args.val_features)
    train_df = pd.read_csv(args.train_manifest)
    val_df = pd.read_csv(args.val_manifest)

    train_ds = VQGDataset(train_df, train_features, vocab, sources=args.sources)
    val_ds = VQGDataset(val_df, val_features, vocab, sources=args.sources)
    collate = make_collate_fn(vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"sources={args.sources}: {len(train_ds)} train examples, {len(val_ds)} val examples, "
          f"vocab={len(vocab)}")

    model = GRNN(vocab_size=len(vocab), pad_id=vocab.pad_id).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    start_epoch, best_val_loss, bad_epochs = 0, float("inf"), 0
    if args.resume and last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=args.device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val_loss = state["best_val_loss"]
        bad_epochs = state["bad_epochs"]
        print(f"Resumed from {last_ckpt} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        train_loss = run_epoch(model, train_loader, vocab, args.device, optimizer)
        val_loss = run_epoch(model, val_loader, vocab, args.device, optimizer=None)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss, bad_epochs = val_loss, 0
            torch.save({"model": model.state_dict(), "vocab": vocab.token2id, "sources": args.sources},
                       best_ckpt)
        else:
            bad_epochs += 1

        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_val_loss": best_val_loss, "bad_epochs": bad_epochs,
        }, last_ckpt)

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    print(f"Best val loss: {best_val_loss:.4f} -- best checkpoint at {best_ckpt}")


if __name__ == "__main__":
    main()
