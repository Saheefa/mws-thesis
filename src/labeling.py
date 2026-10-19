"""Stage 2: weak labeling of the unlabeled Train pool with three sources.

Each labeling function (LF) returns one vote per headline: a class id, or ABSTAIN (-1).
  1. keyword   - count the extracted keywords of each class; vote for the class with most hits.
  2. regex     - patterns built from the keywords ("starts with kw", "kw1 ... kw2"), kept only if
                 they are precise on Dev; vote for the class with most matching patterns.
  3. ds_cluster- distant supervision clustering: TF-IDF similarity to the labeled Dev headlines
                 (the knowledge base). Vote for the most similar class if similarity >= tau.
Ties and zero evidence -> ABSTAIN.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalise, tokenize

ABSTAIN = -1
LF_NAMES = ["keyword", "regex", "ds_cluster"]


def _vote(scores: dict) -> int:
    best = max(scores.values())
    if best <= 0:
        return ABSTAIN
    top = [c for c, s in scores.items() if s == best]
    return top[0] if len(top) == 1 else ABSTAIN


# ---------- 1. keyword matching ----------
def lf_keyword(texts, keywords: dict) -> np.ndarray:
    kw = {c: set(ws) for c, ws in keywords.items()}
    out = []
    for t in texts:
        toks = set(tokenize(t))
        out.append(_vote({c: len(toks & ws) for c, ws in kw.items()}))
    return np.array(out)


# ---------- 2. regular expressions ----------
def build_patterns(dev_df: pd.DataFrame, keywords: dict, min_prec: float = 0.8,
                   min_support: int = 3) -> dict:
    """Candidate patterns per class from its keywords, filtered by precision/support on Dev."""
    norm = dev_df["title"].map(normalise)
    labels = dev_df["label"].to_numpy()
    out = {}
    for c, ws in keywords.items():
        cands = [rf"^{re.escape(w)}\b" for w in ws]
        cands += [rf"\b{re.escape(a)}\b.*\b{re.escape(b)}\b" for a, b in itertools.permutations(ws, 2)]
        keep = []
        for p in cands:
            hit = norm.str.contains(p, regex=True).to_numpy()
            if hit.sum() >= min_support and (labels[hit] == c).mean() >= min_prec:
                keep.append(p)
        out[c] = keep
    return out


def lf_regex(texts, patterns: dict) -> np.ndarray:
    comp = {c: [re.compile(p) for p in ps] for c, ps in patterns.items()}
    out = []
    for t in texts:
        n = normalise(t)
        out.append(_vote({c: sum(bool(r.search(n)) for r in rs) for c, rs in comp.items()}))
    return np.array(out)


# ---------- 3. distant supervision clustering ----------
class DSCluster:
    """Similarity to the Dev knowledge base. Score for class c = mean of the k highest cosine
    similarities between the headline and Dev headlines of class c."""

    def __init__(self, dev_df: pd.DataFrame, fit_texts, k: int = 3):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=2)
        self.vec.fit(list(fit_texts))  # unlabeled text only (Dev + Train titles)
        self.classes = sorted(dev_df["label"].unique())
        self.kb = {c: self.vec.transform(dev_df.loc[dev_df["label"] == c, "title"]) for c in self.classes}
        self.k = k

    def scores(self, texts) -> np.ndarray:
        X = self.vec.transform(list(texts))
        cols = []
        for c in self.classes:
            sim = (X @ self.kb[c].T).toarray()
            k = min(self.k, sim.shape[1])
            cols.append(np.sort(sim, axis=1)[:, -k:].mean(axis=1))
        return np.column_stack(cols)

    def lf(self, texts, tau: float) -> np.ndarray:
        s = self.scores(texts)
        best = s.argmax(axis=1)
        top = s.max(axis=1)
        tie = (s == top[:, None]).sum(axis=1) > 1
        votes = np.array(self.classes)[best]
        return np.where((top >= tau) & ~tie, votes, ABSTAIN)


# ---------- label matrix ----------
def label_matrix(texts, keywords, patterns, ds: DSCluster, tau: float) -> np.ndarray:
    texts = list(texts)
    return np.column_stack([lf_keyword(texts, keywords), lf_regex(texts, patterns), ds.lf(texts, tau)])


def lf_summary(L: np.ndarray, y: np.ndarray | None = None) -> pd.DataFrame:
    """Coverage / overlap / conflict per LF; accuracy only when gold labels are passed (Dev/Test)."""
    rows = []
    voted = L != ABSTAIN
    for j, name in enumerate(LF_NAMES):
        others = np.delete(L, j, axis=1)
        ov = voted[:, j] & (others != ABSTAIN).any(axis=1)
        cf = voted[:, j] & ((others != ABSTAIN) & (others != L[:, [j]])).any(axis=1)
        r = {"lf": name, "coverage": voted[:, j].mean(), "overlap": ov.mean(), "conflict": cf.mean()}
        if y is not None:
            r["accuracy"] = (L[voted[:, j], j] == y[voted[:, j]]).mean() if voted[:, j].any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows).set_index("lf").round(3)
