"""
dataset.py
---------------------------------
PyTorch Dataset for training the GRNN, pairing the image's precomputed fc7
feature vector with a tokenized/encoded question.

One image, one random reference per epoch -- not all 5 expanded into fixed
parallel examples. With 5 human questions per image genuinely disagreeing
(that diversity is the whole point of VQG), presenting all 5 as equally-
weighted targets for the *same* input in *every* epoch pushes plain
cross-entropy toward the token sequence with the best average likelihood
across all 5 -- i.e. a generic, safe, image-agnostic question -- rather than
committing to any one of them. Sampling fresh each epoch still exposes the
model to the full reference diversity over a training run, but without
forcing it to average over conflicting targets within a single epoch.
"""
import random

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
        self.examples = []  # (image_id, [question_str, ...])
        for row in df.itertuples():
            if row.image_id not in features:
                continue
            questions = [q.strip() for q in str(row.questions).split(" | ") if q.strip()]
            if questions:
                self.examples.append((row.image_id, questions))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_id, questions = self.examples[idx]
        question = random.choice(questions)
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
