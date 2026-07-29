"""
dataset.py
---------------------------------
PyTorch Dataset for training the GRNN. Expands each image into one training
example per reference question (typically 5), pairing the image's precomputed
fc7 feature vector with a tokenized/encoded question.
"""
import torch
from torch.utils.data import Dataset

from .vocab import tokenize


class VQGDataset(Dataset):
    def __init__(self, manifest_df, features, vocab, sources=None, max_len=30):
        df = manifest_df
        if sources:
            df = df[df["source"].isin(sources)]
        self.vocab = vocab
        self.features = features
        self.max_len = max_len
        self.examples = []  # (image_id, question_str)
        for row in df.itertuples():
            if row.image_id not in features:
                continue
            for q in str(row.questions).split(" | "):
                q = q.strip()
                if q:
                    self.examples.append((row.image_id, q))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_id, question = self.examples[idx]
        feat = self.features[image_id]
        tokens = tokenize(question)[: self.max_len]
        ids = self.vocab.encode(tokens)
        decoder_input = [self.vocab.sos_id] + ids
        target = ids + [self.vocab.eos_id]
        return feat, torch.tensor(decoder_input, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def make_collate_fn(pad_id):
    def collate_fn(batch):
        feats, inputs, targets = zip(*batch)
        feats = torch.stack(feats)
        max_len = max(len(t) for t in inputs)
        padded_inputs = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        padded_targets = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            padded_inputs[i, : len(inp)] = inp
            padded_targets[i, : len(tgt)] = tgt
        return feats, padded_inputs, padded_targets

    return collate_fn
