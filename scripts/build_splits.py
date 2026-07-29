"""
build_splits.py
---------------------------------
Merges the per-source cleaned VQG CSVs (bing/coco/flickr x train/val/test,
copied into data/raw/ from Visual_Question_Generation_dataset_1.0/cleaned/)
into:
    data/raw/<source>_<split>.csv   normalized per-source file (adds `source` col)
    data/raw/all_<split>.csv         all three sources concatenated (for GRNN_all)

Input files are expected as data/raw/<source>_<split>_all_clean.csv.

Note on the `questions` field: clean_vqg_dataset.py's docstring claims it joins
multi-question cells with " | ", but its parse_questions() separator list never
actually included "---", which is what the raw dataset really uses between the
~5 questions per image -- so cells came through as one unsplit "---"-joined
blob. This script re-splits on "---" here (falling back to the cell as-is if
there's only one question) and re-joins with " | ", which becomes this repo's
canonical in-file separator from this point on.

Usage:
    python scripts/build_splits.py --dir data/raw
"""
import argparse
from pathlib import Path

import pandas as pd

SOURCES = ["bing", "coco", "flickr"]
SPLITS = ["train", "val", "test"]
COMMON_COLS = ["image_id", "source", "image_url", "questions", "source_type"]


def resplit_questions(cell):
    if pd.isna(cell):
        return ""
    parts = [p.strip() for p in str(cell).split("---") if p.strip()]
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw", help="Folder with the *_all_clean.csv files")
    args = ap.parse_args()
    base = Path(args.dir)

    for split in SPLITS:
        per_source_frames = []
        for source in SOURCES:
            in_path = base / f"{source}_{split}_all_clean.csv"
            if not in_path.exists():
                print(f"[MISSING] {in_path}")
                continue
            df = pd.read_csv(in_path)
            df["source"] = source
            df["questions"] = df["questions"].apply(resplit_questions)
            df = df[df["questions"] != ""].reset_index(drop=True)
            if "source_type" not in df.columns:
                df["source_type"] = "unknown"
            out_cols = [c for c in COMMON_COLS if c in df.columns]
            normalized = df[out_cols].copy()

            norm_path = base / f"{source}_{split}.csv"
            normalized.to_csv(norm_path, index=False)
            per_source_frames.append(normalized)
            print(f"{source}_{split}: {len(normalized)} rows -> {norm_path.name}")

        if per_source_frames:
            merged = pd.concat(per_source_frames, ignore_index=True)
            merged_path = base / f"all_{split}.csv"
            merged.to_csv(merged_path, index=False)
            print(f"all_{split}: {len(merged)} rows -> {merged_path.name}")


if __name__ == "__main__":
    main()
