"""
vocab.py
---------------------------------
Tokenization and Vocabulary for VQG questions.

Tokenization is deliberately simple (lowercase, split words from punctuation)
and used identically everywhere -- vocab construction, model input encoding,
and evaluation references -- so nothing pipeline-side diverges from what the
model actually sees.
"""
import json
import re
from collections import Counter
from pathlib import Path

PAD, SOS, EOS, UNK = "<pad>", "<sos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, SOS, EOS, UNK]

# Keeps contractions ("don't", "it's") as single tokens instead of fragmenting
# them into a bare word plus a stray apostrophe plus a meaningless single letter.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[?.!,;:]")


def tokenize(text):
    text = str(text).strip().lower()
    return _TOKEN_RE.findall(text)


class Vocab:
    def __init__(self, token2id):
        self.token2id = token2id
        self.id2token = {i: t for t, i in token2id.items()}

    def __len__(self):
        return len(self.token2id)

    @property
    def pad_id(self):
        return self.token2id[PAD]

    @property
    def sos_id(self):
        return self.token2id[SOS]

    @property
    def eos_id(self):
        return self.token2id[EOS]

    @property
    def unk_id(self):
        return self.token2id[UNK]

    def encode(self, tokens):
        unk = self.unk_id
        return [self.token2id.get(t, unk) for t in tokens]

    def decode(self, ids, stop_at_eos=True):
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            out.append(self.id2token.get(i, UNK))
        return out

    @classmethod
    def build(cls, questions, min_freq=3):
        counter = Counter()
        for q in questions:
            counter.update(tokenize(q))
        # Paper's rule: keep every word seen >=3 times in training (Sec 4.1).
        kept = sorted(t for t, c in counter.items() if c >= min_freq)
        token2id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for t in kept:
            if t not in token2id:
                token2id[t] = len(token2id)
        return cls(token2id)

    def save(self, path):
        Path(path).write_text(json.dumps(self.token2id, indent=2))

    @classmethod
    def load(cls, path):
        token2id = json.loads(Path(path).read_text())
        return cls(token2id)
