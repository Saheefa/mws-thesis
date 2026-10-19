"""Shared text normalisation for every pipeline stage.

Design choice: only a SMALL stop list (articles, prepositions, conjunctions, forms of "be").
Pronouns and question words ("you", "this", "which", "why") are kept on purpose, because
they are strong clickbait cues; a full stop list would delete them. Every number becomes
the token NUM ("19 Food Trucks" -> "NUM food trucks"), so list-style headlines share a token.
"""
from __future__ import annotations

import re

NUM = "num"

STOP = frozenset(
    """a an the of in on at to for from by with as into onto over under about after before
    and or but nor so if than then
    is are was were be been being am
    it its s t""".split()
)

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*")


def tokenize(text: str, drop_stop: bool = True) -> list[str]:
    toks = []
    for t in _TOKEN.findall(str(text).lower()):
        if t[0].isdigit():
            t = NUM
        elif "'" in t:
            t = t.split("'")[0] if t.endswith(("'s", "'t", "'re", "'ll", "'ve", "'d", "'m")) else t
        if drop_stop and t in STOP:
            continue
        toks.append(t)
    return toks


def normalise(text: str) -> str:
    """Token string used by regex labeling functions (stop words kept, numbers -> NUM)."""
    return " ".join(tokenize(text, drop_stop=False))
