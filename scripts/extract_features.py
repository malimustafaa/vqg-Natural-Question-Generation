"""
extract_features.py
---------------------------------
Runs a frozen, ImageNet-pretrained VGG16 over every downloaded image and
saves its 4096-d fc7 feature vector (Section 4.1: "the 4096-dimensional
output of the last fully connected layer (fc7)"). In the classic Caffe VGG16
naming (which the paper uses), fc7 is the SECOND fully-connected layer
(fc6 -> relu -> fc7 -> relu), i.e. torchvision's classifier[0:5] in eval mode
(dropout is a no-op then). The final 1000-way classifier (fc8) is discarded
entirely -- it's never used by the GRNN.

Features are precomputed once per split and cached to
data/features/<split>.pt (dict: image_id -> 4096-d tensor) so training never
re-runs the (frozen, non-backprop'd) CNN -- this is the practical equivalent
of the paper's "we do not back-propagate the CNN".

Usage:
    python scripts/extract_features.py --manifest data/raw/manifest_train.csv \
        --out data/features/train.pt
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import VGG16_Weights, vgg16
from tqdm import tqdm

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VGG16Fc7(nn.Module):
    """Frozen VGG16 truncated to its fc7 (second FC + ReLU) output, 4096-d."""

    def __init__(self, pretrained=True):
        super().__init__()
        base = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.fc7 = nn.Sequential(*list(base.classifier.children())[:5])  # fc6,relu,drop,fc7,relu
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc7(x)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return IMAGENET_TRANSFORM(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-pretrained", action="store_true",
                     help="Skip downloading ImageNet weights -- random-init CNN, for smoke-testing "
                          "the pipeline's plumbing (shapes, no crashes) on a slow connection. Never "
                          "use this for real feature extraction.")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    model = VGG16Fc7(pretrained=not args.no_pretrained).to(args.device)

    features = {}
    skipped = []
    ids, paths = df["image_id"].tolist(), df["image_path"].tolist()

    for start in tqdm(range(0, len(ids), args.batch_size), desc="extracting fc7"):
        batch_ids = ids[start : start + args.batch_size]
        batch_paths = paths[start : start + args.batch_size]
        tensors, kept_ids = [], []
        for img_id, path in zip(batch_ids, batch_paths):
            try:
                tensors.append(load_image(path))
                kept_ids.append(img_id)
            except Exception as e:
                skipped.append((img_id, str(e)))
        if not tensors:
            continue
        batch = torch.stack(tensors).to(args.device)
        feats = model(batch).cpu()
        for img_id, feat in zip(kept_ids, feats):
            features[img_id] = feat

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.out)
    print(f"Extracted {len(features)} feature vectors -> {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)} unreadable images (corrupt download): {skipped[:5]}")


if __name__ == "__main__":
    main()
