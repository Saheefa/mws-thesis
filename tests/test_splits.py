"""Sanity checks for the split logic. Runs on a tiny synthetic table (test fixture only,
NOT experiment data):  python tests/test_splits.py   (or pytest)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import data


def _fixture(n_per_class: int = 500) -> pd.DataFrame:
    rows = [(f"clickbait fixture headline number {i}", 1) for i in range(n_per_class)]
    rows += [(f"ordinary fixture headline number {i}", 0) for i in range(n_per_class)]
    rows += [("  Clickbait   fixture headline number 0 ", 1)]  # whitespace/case duplicate
    rows += [("conflict title", 1), ("Conflict Title", 0)]  # appears in both classes
    return pd.DataFrame(rows, columns=["title", "label"])


def test_clean_and_split():
    df = data.clean(_fixture())
    assert len(df) == 1000, len(df)  # duplicate and conflicting titles removed
    sp = data.make_splits(df, dev_per_class=50, test_frac=0.2, seed=1)
    dev, train, test = sp["dev"], sp["train"], sp["test"]
    assert len(dev) == 100 and dev["label"].sum() == 50
    assert len(test) == 200
    assert len(train) == 1000 - 100 - 200
    assert "label" not in train.columns and "y_hidden" in train.columns
    sets = [set(x["title"].str.lower()) for x in (dev, train, test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert abs(train["y_hidden"].mean() - 0.5) < 0.02


def test_reproducible_and_subsample():
    df = data.clean(_fixture())
    a = data.make_splits(df, 50, 0.2, seed=7)
    b = data.make_splits(df, 50, 0.2, seed=7)
    assert a["dev"]["title"].tolist() == b["dev"]["title"].tolist()
    small = data.subsample_train(a["train"], 200, seed=7)
    assert len(small) == 200 and abs(small["y_hidden"].mean() - 0.5) < 0.05


if __name__ == "__main__":
    test_clean_and_split()
    test_reproducible_and_subsample()
    print("all split tests passed")
