"""
model.py
---------------------------------
GRNN: the generative model from Mostafazadeh et al. 2016, Section 4.1.

    fc7 (4096-d, frozen VGG16 features) --Linear--> h0 (500-d)
    h0 seeds a single-layer GRU (hidden size 500) that generates the
    question one word at a time, embedding size 500, until <eos>.

The CNN itself is not part of this module -- features are precomputed by
scripts/extract_features.py (paper: "we do not back-propagate the CNN"), so
this model takes fc7 vectors as input, not raw images.

Two details the paper states only loosely -- both flagged in README.md:
  - word embedding dim: 500 (matches GRU input size; not stated explicitly)
  - fc7->500 bridge: plain Linear, no activation (paper just says "transform")
"""
import torch.nn as nn

FC7_DIM = 4096
HIDDEN_DIM = 500
EMBED_DIM = 500


class GRNN(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.bridge = nn.Linear(FC7_DIM, HIDDEN_DIM)
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=pad_id)
        self.gru = nn.GRU(EMBED_DIM, HIDDEN_DIM, num_layers=1, batch_first=True)
        self.output_layer = nn.Linear(HIDDEN_DIM, vocab_size)

    def init_hidden(self, fc7_features):
        """fc7_features: (batch, 4096) -> (1, batch, 500), the GRU's h0."""
        return self.bridge(fc7_features).unsqueeze(0)

    def forward(self, fc7_features, input_tokens):
        """
        Teacher-forced training step.
        fc7_features: (batch, 4096)
        input_tokens: (batch, seq_len) -- decoder inputs, i.e. <sos> + question
        Returns logits: (batch, seq_len, vocab_size)
        """
        h0 = self.init_hidden(fc7_features)
        embedded = self.embedding(input_tokens)
        output, _ = self.gru(embedded, h0)
        return self.output_layer(output)

    def step(self, token, hidden):
        """Single decoding step for beam search.
        token: (batch,) int64. hidden: (1, batch, 500).
        Returns logits (batch, vocab_size) and the updated hidden state.
        """
        embedded = self.embedding(token).unsqueeze(1)  # (batch, 1, embed)
        output, hidden = self.gru(embedded, hidden)
        logits = self.output_layer(output.squeeze(1))  # (batch, vocab)
        return logits, hidden
