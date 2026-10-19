"""Stage 3 (label integration) and stage 4 (soft-label classifier)."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from snorkel.labeling.model import LabelModel
from torch import nn

from .labeling import ABSTAIN
from .metrics import macro_f1
from .text import tokenize

SEED = 42


# ---------- stage 3: Snorkel label model ----------
def fit_label_model(L: np.ndarray, cardinality: int, seed: int = SEED) -> np.ndarray:
    """Returns probabilistic labels (n x k). Rows where every LF abstains are NOT dropped here."""
    lm = LabelModel(cardinality=cardinality, verbose=False)
    lm.fit(L_train=L, n_epochs=500, lr=0.01, log_freq=100, seed=seed)
    return lm.predict_proba(L)


def covered(L: np.ndarray) -> np.ndarray:
    return (L != ABSTAIN).any(axis=1)


# ---------- stage 4: bag-of-words NN trained on soft labels ----------
class BowNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def _to_tensor(X):
    return torch.tensor(X.toarray(), dtype=torch.float32)


class SoftLabelClassifier:
    """CountVectorizer (binary BoW) + 1-hidden-layer NN, loss = cross-entropy vs soft targets.
    The epoch with the best macro-F1 on Dev is kept (Dev is labeled, so this is allowed)."""

    def __init__(self, n_classes: int, epochs: int = 15, batch: int = 64, lr: float = 1e-3,
                 seed: int = SEED):
        self.k, self.epochs, self.batch, self.lr, self.seed = n_classes, epochs, batch, lr, seed

    def fit(self, texts, soft_y: np.ndarray, dev_texts=None, dev_y=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   binary=True, min_df=2)
        X = _to_tensor(self.vec.fit_transform(list(texts)))
        Y = torch.tensor(soft_y, dtype=torch.float32)
        self.model = BowNet(X.shape[1], self.k)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best, best_state, self.history = -1.0, None, []
        for ep in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(X))
            for i in range(0, len(X), self.batch):
                idx = order[i:i + self.batch]
                logp = torch.log_softmax(self.model(X[idx]), dim=1)
                loss = -(Y[idx] * logp).sum(dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if dev_texts is not None:
                f = macro_f1(dev_y, self.predict(dev_texts))
                self.history.append(f)
                if f > best:
                    best, best_state = f, {k: v.clone() for k, v in self.model.state_dict().items()}
                    self.best_epoch = ep + 1
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(_to_tensor(self.vec.transform(list(texts)))), dim=1).numpy()

    def predict(self, texts) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)
