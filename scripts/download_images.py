"""
download_images.py
---------------------------------
Downloads VQG images from data/raw/<source>_<split>.csv (image_url column)
into data/images/<source>/<split>/<image_id>.<ext>, and writes a manifest CSV
(data/raw/manifest_<split>.csv) listing only the images that actually
downloaded successfully -- consumed downstream by extract_features.py.

Some of clean_vqg_dataset.py's "working" links were verified alive at
link-check time but can still fail at actual download time (transient errors,
images removed since). This script re-verifies by actually saving bytes, not
just trusting the earlier HEAD/GET check. It's resumable: already-downloaded
files are skipped, so interrupting and re-running is safe.

Usage:
    # tiny local dev sample (default limit keeps this fast on a laptop)
    python scripts/download_images.py --dir data --limit 20

    # full-scale download (run this on Colab, --dir pointed at Drive)
    python scripts/download_images.py --dir data
"""
import argparse
import concurrent.futures as cf
import csv
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

SOURCES = ["bing", "coco", "flickr"]
SPLITS = ["train", "val", "test"]

EXT_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
}


def guess_ext(url, content_type):
    ext = EXT_BY_CONTENT_TYPE.get((content_type or "").lower())
    if ext:
        return ext
    ext = Path(str(url).split("?")[0]).suffix.lower()
    return ext if ext in (".jpg", ".jpeg", ".png", ".gif") else ".jpg"


def download_one(image_id, url, out_dir, timeout):
    existing = list(out_dir.glob(f"{image_id}.*"))
    if existing:
        return image_id, str(existing[0]), "already_downloaded"
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        if r.status_code != 200:
            return image_id, None, f"http_{r.status_code}"
        content = r.content
        if len(content) < 512:
            return image_id, None, "too_small"
        ext = guess_ext(url, r.headers.get("Content-Type"))
        out_path = out_dir / f"{image_id}{ext}"
        out_path.write_bytes(content)
        return image_id, str(out_path), "ok"
    except requests.exceptions.RequestException as e:
        return image_id, None, f"error:{type(e).__name__}"


def download_split_source(df, out_dir, workers, timeout, desc):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one, row.image_id, row.image_url, out_dir, timeout): row.image_id
            for row in df.itertuples()
        }
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc=desc):
            image_id, path, status = fut.result()
            results[image_id] = (path, status)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data", help="Base data dir (expects <dir>/raw, writes <dir>/images)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None,
                     help="Max images per source per split (omit for full-scale download)")
    ap.add_argument("--splits", nargs="*", default=SPLITS)
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    args = ap.parse_args()

    base = Path(args.dir)
    raw_dir = base / "raw"
    images_dir = base / "images"

    summary = []
    for split in args.splits:
        manifest_rows = []
        for source in args.sources:
            csv_path = raw_dir / f"{source}_{split}.csv"
            if not csv_path.exists():
                print(f"[MISSING] {csv_path} -- run build_splits.py first")
                continue
            df = pd.read_csv(csv_path)
            if args.limit:
                df = df.head(args.limit)

            out_dir = images_dir / source / split
            results = download_split_source(
                df, out_dir, args.workers, args.timeout, desc=f"{source}/{split}"
            )

            n_ok = 0
            for row in df.itertuples():
                path, status = results.get(row.image_id, (None, "not_attempted"))
                if path:
                    n_ok += 1
                    manifest_rows.append({
                        "image_id": row.image_id,
                        "source": source,
                        "image_path": path,
                        "questions": row.questions,
                    })
            summary.append({
                "source": source, "split": split,
                "attempted": len(df), "downloaded": n_ok,
                "pct": round(100 * n_ok / max(len(df), 1), 1),
            })
            print(f"{source}/{split}: {n_ok}/{len(df)} downloaded")

        if manifest_rows:
            manifest_path = raw_dir / f"manifest_{split}.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"manifest_{split}: {len(manifest_rows)} rows -> {manifest_path}")

    if summary:
        report = pd.DataFrame(summary)
        report_path = raw_dir / "download_report.csv"
        report.to_csv(report_path, index=False)
        print("\n=== Download Summary ===")
        print(report.to_string(index=False))


if __name__ == "__main__":
    main()
