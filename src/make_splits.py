"""CLI: build Dev/Train/Test splits from the raw Clickbait files.

Usage (from the repo root):
    python -m src.make_splits --cb data/raw/clickbait_data --ncb data/raw/non_clickbait_data
"""
from __future__ import annotations

import argparse

from . import data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cb", required=True, help="clickbait headlines file (one per line)")
    p.add_argument("--ncb", required=True, help="non-clickbait headlines file (one per line)")
    p.add_argument("--out", default="data/processed/clickbait")
    p.add_argument("--dev-per-class", type=int, default=100)
    p.add_argument("--test-frac", type=float, default=0.20)
    p.add_argument("--small-train", type=int, default=2000, help="size of the small-scale Train variant")
    p.add_argument("--seed", type=int, default=data.SEED)
    a = p.parse_args()

    df = data.load_clickbait(a.cb, a.ncb)
    splits = data.make_splits(df, a.dev_per_class, a.test_frac, a.seed)
    splits_small = dict(splits, train=data.subsample_train(splits["train"], a.small_train, a.seed))
    data.save_splits(splits, a.out + "/large")
    data.save_splits(splits_small, a.out + "/small")

    print(f"cleaned total: {len(df)}  (clickbait share {df['label'].mean():.3f})")
    for scale, sp in (("large", splits), ("small", splits_small)):
        for name, frame in sp.items():
            col = "y_hidden" if name == "train" else "label"
            print(f"[{scale}] {name:5s} n={len(frame):6d}  clickbait share={frame[col].mean():.3f}")


if __name__ == "__main__":
    main()
