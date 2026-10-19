"""Data loading and Dev/Train/Test splitting for the MWS thesis experiments.

Convention: label 1 = clickbait, label 0 = non-clickbait.

Split design (matches the thesis proposal):
  * Dev   - small labeled set. Used ONLY for knowledge extraction, threshold
            tuning and the distant-supervision knowledge base.
  * Train - large pool. Ground-truth labels are kept in `y_hidden` and must
            never be read by the pipeline (simulates the weakly supervised setting).
  * Test  - held-out labeled set, used only for final evaluation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42


def _read_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_clickbait(clickbait_path: str | Path, non_clickbait_path: str | Path) -> pd.DataFrame:
    """Load the two Chakraborty et al. (2016) files into one `title,label` table."""
    df = pd.concat(
        [
            pd.DataFrame({"title": _read_lines(clickbait_path), "label": 1}),
            pd.DataFrame({"title": _read_lines(non_clickbait_path), "label": 0}),
        ],
        ignore_index=True,
    )
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise whitespace, drop empties/duplicates, drop titles that occur in both classes."""
    df = df.copy()
    df["title"] = df["title"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["title"].str.len() > 0]
    key = df["title"].str.lower()
    conflicting = set(key[df.assign(k=key).groupby("k")["label"].transform("nunique") > 1])
    df = df[~key.isin(conflicting)]
    df = df.assign(_k=df["title"].str.lower()).drop_duplicates("_k").drop(columns="_k")
    return df.reset_index(drop=True)


def make_splits(
    df: pd.DataFrame,
    dev_per_class: int = 100,
    test_frac: float = 0.20,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Stratified, seeded Dev / Train / Test split.

    Dev has `dev_per_class` examples of each class (so 200 for the binary case).
    Test is `test_frac` of the full data. Train is everything else, with the
    ground truth moved to `y_hidden`.
    """
    rest, test = train_test_split(df, test_size=test_frac, stratify=df["label"], random_state=seed)
    dev = rest.groupby("label", group_keys=False).sample(n=dev_per_class, random_state=seed)
    train = rest.drop(index=dev.index)
    train = train.rename(columns={"label": "y_hidden"})
    return {
        "dev": dev.reset_index(drop=True),
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def subsample_train(train: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    """Stratified subsample of the Train pool, used for the small-scale variant."""
    if n >= len(train):
        return train.copy()
    out, _ = train_test_split(train, train_size=n, stratify=train["y_hidden"], random_state=seed)
    return out.reset_index(drop=True)


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(out / f"{name}.csv", index=False)


def load_pipeline_splits(in_dir: str | Path) -> dict[str, pd.DataFrame]:
    """What the weak-supervision pipeline is allowed to see.

    Train is read with `usecols=["title"]`, so `y_hidden` is never even loaded into memory.
    """
    d = Path(in_dir)
    return {
        "dev": pd.read_csv(d / "dev.csv"),
        "train": pd.read_csv(d / "train.csv", usecols=["title"]),
        "test": pd.read_csv(d / "test.csv"),
    }


def load_splits(in_dir: str | Path) -> dict[str, pd.DataFrame]:
    d = Path(in_dir)
    return {n: pd.read_csv(d / f"{n}.csv") for n in ("dev", "train", "test")}
