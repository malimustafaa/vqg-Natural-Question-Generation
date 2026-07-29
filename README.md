# VQG 2016 Reproduction

Faithful reproduction of the generative model (**GRNN**) from:

> Nasrin Mostafazadeh, Ishan Misra, Jacob Devlin, Margaret Mitchell, Xiaodong He, Lucy Vanderwende.
> **"Generating Natural Questions About an Image."** ACL 2016. https://aclanthology.org/P16-1170/

Given an image, generate a natural, engaging question about it (not a literal, visually-verifiable one).

## Architecture (Section 4.1 of the paper, followed exactly)

- **CNN**: VGGNet, ImageNet-pretrained, frozen (never backpropagated). Uses the 4096-dim `fc7` layer output.
- **Bridge**: `fc7` (4096) → `Linear` → 500-dim vector, used as the GRU's **initial hidden state** (the image
  is not re-fed at every decoding step — it only seeds `h0`).
- **Decoder**: single-layer **GRU**, hidden size 500. Produces one word at a time until `<eos>`.
- **Training**: GRU + bridge matrix trained jointly with **SGD + early stopping**.
- **Vocab**: all words with frequency ≥3 in the training questions. `<unk>` used during training, but the
  decoder is not allowed to emit it at test time.
- **Decoding**: beam search, width 8.
- Two regimes, both reproduced: `GRNN_all` (trained on coco+flickr+bing pooled) and `GRNN_X` (trained
  per-source: coco-only / flickr-only / bing-only).

### Two details the paper doesn't state explicitly

The paper describes the architecture at a level that leaves two implementation choices unstated. Both are
called out here rather than silently assumed:

1. **Word embedding dimension** — not given. Implemented as 500, to match the GRU's input size (the natural
   reading, and consistent with the contemporaneous show-and-tell-style captioning models this architecture
   is based on).
2. **`fc7 → 500` bridge nonlinearity** — the paper just says "transform." Implemented as a plain `Linear`
   layer with no activation. See `vqg/model.py`.

## Known deviations from the paper (data availability, not architecture)

The original dataset's image URLs are from 2016; many are dead now. After link-checking + Wayback Machine
recovery (see `Visual_Question_Generation_dataset_1.0/cleaned/` in this project's source data):

| source | usable images | paper's original |
|---|---|---|
| coco   | 4,990 | 5,000 |
| flickr | 4,135 | 5,000 |
| bing   | 1,926 | 5,000 (2016 Bing image-search result URLs are largely unrecoverable) |

Consequences:
- Our vocabulary size will differ from the paper's reported 1,942 tokens (smaller training corpus at the
  same freq≥3 threshold).
- We report **BLEU** (4-gram, equal weights) and **METEOR** but not **∆BLEU** — ∆BLEU requires crowdsourcing
  3 human quality ratings per reference question, which is out of scope here. The paper itself notes BLEU is
  a strong standalone proxy for ∆BLEU when per-reference ratings aren't available.
- SGD learning rate / batch size / epoch count aren't specified in the paper. They're CLI flags in
  `scripts/train.py` with reasonable defaults, meant to be tuned against validation BLEU with early stopping.

## Pipeline

```
scripts/build_splits.py      # merge per-source cleaned CSVs -> data/raw/all_{train,val,test}.csv
scripts/build_vocab.py       # freq>=3 vocab from train questions -> data/raw/vocab.json
scripts/download_images.py   # image_url -> data/images/<source>/<split>/<image_id>.jpg
scripts/extract_features.py  # frozen VGG16 fc7 -> data/features/<split>.pt (image_id -> 4096-d tensor)
scripts/train.py             # --sources all|coco|flickr|bing
scripts/decode.py            # beam search inference
scripts/evaluate.py          # BLEU + METEOR, Table-5-style report
```

## Workflow

Everything is written and CPU-smoke-tested locally (on a tiny image sample) in this repo, then pushed to
GitHub. Full-scale image download + GPU training happens in `notebooks/colab_train.ipynb` on Colab Pro+,
with Google Drive as persistent storage for the full image set and checkpoints (nothing large is committed
to this repo — see `.gitignore`).
