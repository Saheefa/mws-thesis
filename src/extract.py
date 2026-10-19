"""Stage 1: knowledge extraction from the labeled Dev set.

Every extractor has the same interface:  extract(dev_df, cls) -> list[str] (ranked keywords).
`dev_df` must have columns `title` and `label`. Extractors only ever see Dev.

After extraction, `discriminative(...)` (shared by ALL methods, so the comparison is fair)
drops words that are about as common in the other classes as in `cls` on Dev.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

from .text import tokenize

SEED = 42


class LDAExtractor:
    """Baseline from Chen et al. (2022): per-class LDA on Dev, keywords = top topic words.

    LDA is fitted separately on each class's Dev headlines. Each word gets a score equal to
    the sum over topics of P(topic) * P(word | topic); the highest-scoring words are returned.
    """

    name = "lda"

    def __init__(self, num_topics: int = 5, topn: int = 15, passes: int = 50, seed: int = SEED):
        self.num_topics, self.topn, self.passes, self.seed = num_topics, topn, passes, seed

    def extract(self, dev_df: pd.DataFrame, cls) -> list[str]:
        docs = [tokenize(t) for t in dev_df.loc[dev_df["label"] == cls, "title"]]
        docs = [d for d in docs if d]
        vocab = corpora.Dictionary(docs)
        vocab.filter_extremes(no_below=2, no_above=1.0, keep_n=None)  # drop words seen in only 1 headline
        bow = [vocab.doc2bow(d) for d in docs]
        lda = LdaModel(
            bow, id2word=vocab, num_topics=self.num_topics, passes=self.passes,
            iterations=200, alpha="auto", eta="auto", random_state=self.seed,
        )
        topic_weight = Counter()
        for doc in bow:
            for k, p in lda.get_document_topics(doc, minimum_probability=0.0):
                topic_weight[k] += p
        total = sum(topic_weight.values()) or 1.0
        score = Counter()
        for k in range(self.num_topics):
            for wid, p in lda.get_topic_terms(k, topn=len(vocab)):
                score[vocab[wid]] += topic_weight[k] / total * p
        return [w for w, _ in score.most_common(self.topn * 4)]  # over-generate; filter trims


def discriminative(dev_df: pd.DataFrame, words: list[str], cls, min_ratio: float = 2.0,
                   topn: int = 15) -> list[str]:
    """Keep words whose Dev document frequency in `cls` is >= min_ratio x that in other classes
    (add-one smoothed, per-class normalised). Returns at most `topn` words, original order kept."""
    in_cls = dev_df["label"] == cls
    df_in = Counter(w for t in dev_df.loc[in_cls, "title"] for w in set(tokenize(t)))
    df_out = Counter(w for t in dev_df.loc[~in_cls, "title"] for w in set(tokenize(t)))
    n_in, n_out = max(in_cls.sum(), 1), max((~in_cls).sum(), 1)
    keep = [w for w in words if (df_in[w] + 1) / n_in >= min_ratio * (df_out[w] + 1) / n_out]
    return keep[:topn]


EXTRACTORS = {"lda": LDAExtractor}


def class_keywords(dev_df: pd.DataFrame, method: str = "lda", **kw) -> dict:
    ex = EXTRACTORS[method](**kw)
    return {c: discriminative(dev_df, ex.extract(dev_df, c), c) for c in sorted(dev_df["label"].unique())}
