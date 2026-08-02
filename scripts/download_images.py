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
import threading
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

SOURCES = ["bing", "coco", "flickr"]
SPLITS = ["train", "val", "test"]

# Some CDNs (Flickr in particular) throttle/block the bare "python-requests"
# UA and bursts from datacenter IPs (e.g. Colab's) -- a normal browser UA plus
# a couple of retries with backoff clears most of these transient failures.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; vqg-dataset-fetch/1.0)"}
MAX_ATTEMPTS = 4
RETRY_BACKOFF = 2.0
# requests' `timeout` only bounds gaps *between* bytes, not the total transfer
# -- a connection trickling data just under that gap can hang far longer than
# `timeout` would suggest. Bound the whole download wall-clock instead.
MAX_TOTAL_SECONDS = 20.0


class _TotalTimeExceeded(Exception):
    pass


class RateLimiter:
    """Paces request *starts* across all worker threads (not per-thread), and
    lets a 429 push everyone's next allowed request further out -- a global
    circuit breaker instead of each thread backing off independently and
    other threads immediately re-tripping the limit."""

    def __init__(self, min_interval):
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait_turn(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            start_at = max(now, self._next_allowed)
            self._next_allowed = start_at + self.min_interval
        sleep_for = start_at - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)

    def penalize(self, seconds):
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)


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


def download_one(image_id, url, out_dir, timeout, limiter):
    existing = list(out_dir.glob(f"{image_id}.*"))
    if existing:
        return image_id, str(existing[0]), "already_downloaded"

    last_status = "not_attempted"
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_BACKOFF * attempt)
        limiter.wait_turn()
        try:
            r = requests.get(url, timeout=timeout, headers=HEADERS, stream=True)
            if r.status_code != 200:
                last_status = f"http_{r.status_code}"
                if r.status_code == 429:
                    # One 429 means the whole IP is throttled -- pause every
                    # thread, not just this one, or they'll immediately
                    # re-trip it in parallel.
                    limiter.penalize(30.0)
                    continue
                # Not worth retrying a real 404/403 (link is gone), but do
                # retry other transient server errors.
                if r.status_code not in (500, 502, 503, 504):
                    break
                continue
            start = time.monotonic()
            chunks = []
            for chunk in r.iter_content(chunk_size=65536):
                chunks.append(chunk)
                if time.monotonic() - start > MAX_TOTAL_SECONDS:
                    raise _TotalTimeExceeded
            content = b"".join(chunks)
            if len(content) < 512:
                last_status = "too_small"
                continue
            ext = guess_ext(url, r.headers.get("Content-Type"))
            out_path = out_dir / f"{image_id}{ext}"
            out_path.write_bytes(content)
            return image_id, str(out_path), "ok"
        except _TotalTimeExceeded:
            last_status = "error:TotalTimeExceeded"
            continue
        except requests.exceptions.RequestException as e:
            last_status = f"error:{type(e).__name__}"
            continue
    return image_id, None, last_status


def download_split_source(df, out_dir, workers, timeout, desc, min_interval):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    limiter = RateLimiter(min_interval)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(download_one, row.image_id, row.image_url, out_dir, timeout, limiter): row.image_id
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
    ap.add_argument("--flickr-workers", type=int, default=3,
                     help="Separate (lower) concurrency for flickr -- its CDN rate-limits hard")
    ap.add_argument("--flickr-min-interval", type=float, default=0.5,
                     help="Minimum seconds between flickr request starts, across all workers combined")
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
            source_workers = args.flickr_workers if source == "flickr" else args.workers
            min_interval = args.flickr_min_interval if source == "flickr" else 0.0
            results = download_split_source(
                df, out_dir, source_workers, args.timeout, desc=f"{source}/{split}",
                min_interval=min_interval,
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
