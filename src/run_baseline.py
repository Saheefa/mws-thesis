"""Step 2: baseline MWS with LDA keywords, end to end.

    python -m src.run_baseline --scale large      (or --scale small)

Dev -> LDA keywords -> 3 LFs on Train -> Snorkel soft labels -> BoW NN -> Test metrics.
Train ground truth is never loaded (see data.load_pipeline_splits). Test is used only at the end.
A reference row trains the same NN on the 200 labeled Dev headlines only (no weak labels).
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic mode on GPU
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import data
from .extract import class_keywords
from .labeling import DSCluster, build_patterns, label_matrix, lf_summary
from .metrics import report
from .model import SoftLabelClassifier, covered, fit_label_model

NAMES = {0: "non-clickbait", 1: "clickbait"}


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)


def run(scale: str, tau: float, method: str = "lda", seed: int = data.SEED) -> dict:
    seed_all(seed)
    sp = data.load_pipeline_splits(f"data/processed/clickbait/{scale}")
    dev, train, test = sp["dev"], sp["train"], sp["test"]
    assert list(train.columns) == ["title"], "pipeline must not see Train labels"
    k = dev["label"].nunique()

    # 1. knowledge extraction (Dev only)
    keywords = class_keywords(dev, method)
    patterns = build_patterns(dev, keywords)

    # 2. weak labeling of Train
    ds = DSCluster(dev, pd.concat([dev["title"], train["title"]]))
    L = label_matrix(train["title"], keywords, patterns, ds, tau)
    cov = covered(L)

    # 3. label integration
    probs = fit_label_model(L, cardinality=k, seed=seed)

    # 4. soft-label classifier on covered Train rows, epoch picked on Dev
    clf = SoftLabelClassifier(k, seed=seed).fit(
        train["title"][cov], probs[cov], dev["title"], dev["label"].to_numpy())

    # reference: same NN, trained on Dev gold labels only (fixed 15 epochs, no selection)
    seed_all(seed)
    ref = SoftLabelClassifier(k, seed=seed).fit(dev["title"], np.eye(k)[dev["label"].to_numpy()])

    # 5. evaluation on Test (only here)
    y = test["label"].to_numpy()
    return {
        "config": {"dataset": "clickbait", "scale": scale, "method": method, "tau": tau, "seed": seed},
        "keywords": {NAMES[int(c)]: w for c, w in keywords.items()},
        "n_patterns": {NAMES[int(c)]: len(p) for c, p in patterns.items()},
        "train": {
            "n": int(len(train)),
            "covered": int(cov.sum()),
            "coverage": round(float(cov.mean()), 4),
            "lf_summary_unlabeled": lf_summary(L).to_dict(orient="index"),
            "soft_label_class_share": {NAMES[c]: round(float(probs[cov, c].mean()), 4) for c in range(k)},
        },
        "best_epoch": clf.best_epoch,
        "dev_macro_f1_by_epoch": [round(f, 4) for f in clf.history],
        "test": report(y, clf.predict(test["title"]), NAMES),
        "reference_dev_only_test": report(y, ref.predict(test["title"]), NAMES),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["large", "small"], default="large")
    p.add_argument("--tau", type=float, default=0.2)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    res = run(a.scale, a.tau)
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"baseline_lda_clickbait_{a.scale}.json"
    path.write_text(json.dumps(res, indent=2))
    t, r = res["test"], res["reference_dev_only_test"]
    print(f"[{a.scale}] coverage={res['train']['coverage']}  best_epoch={res['best_epoch']}")
    print(pd.DataFrame(t["per_class"]).T.to_string())
    print(f"MWS-LDA   macro-F1={t['macro_f1']}  weighted-F1={t['weighted_f1']}  acc={t['accuracy']}")
    print(f"Dev-only  macro-F1={r['macro_f1']}  weighted-F1={r['weighted_f1']}  acc={r['accuracy']}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
