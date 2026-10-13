"""Multi-layer perceptron in PyTorch, wrapped to behave like a scikit-learn estimator.

The optimizer is a hyperparameter so the syllabus optimizers (GD, SGD, Momentum,
Adagrad, Adam) can be compared on the same network.
"""
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from torch import nn

from src.config import SEED

OPTIMIZERS = {
    # name: (torch optimizer, kwargs, batch size; None = full batch)
    "GD": (torch.optim.SGD, {"lr": 0.05}, None),
    "SGD": (torch.optim.SGD, {"lr": 0.005}, 64),
    "Momentum": (torch.optim.SGD, {"lr": 0.005, "momentum": 0.9}, 64),
    "Adagrad": (torch.optim.Adagrad, {"lr": 0.01}, 64),
    "Adam": (torch.optim.Adam, {"lr": 1e-3}, 64),
}


def make_mlp(n_in, hidden, dropout):
    layers, width = [], n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class _TorchMLP(BaseEstimator):
    """Shared training loop. The last `val_frac` of the (time-ordered) training data
    is held out for early stopping, so the model never peeks at later seasons."""

    def __init__(self, hidden=(64, 32), dropout=0.2, weight_decay=1e-3, optimizer="Adam",
                 epochs=300, patience=25, val_frac=0.1, seed=SEED):
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed

    def _prepare_targets(self, y):
        raise NotImplementedError

    def fit(self, X, y, eval_set=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = self._prepare_targets(np.asarray(y, dtype=np.float32))
        if eval_set is None:
            cut = int(len(X) * (1 - self.val_frac))
            X, Xv, y, yv = X[:cut], X[cut:], y[:cut], y[cut:]
        else:
            Xv = np.asarray(eval_set[0], dtype=np.float32)
            yv = self._prepare_targets(np.asarray(eval_set[1], dtype=np.float32))

        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        Xvt, yvt = torch.from_numpy(Xv), torch.from_numpy(yv)
        self.model_ = make_mlp(X.shape[1], self.hidden, self.dropout)
        opt_cls, opt_kwargs, batch = OPTIMIZERS[self.optimizer]
        opt = opt_cls(self.model_.parameters(), weight_decay=self.weight_decay, **opt_kwargs)
        batch = batch or len(Xt)

        self.history_ = {"train": [], "val": []}
        best, best_state, stale = np.inf, None, 0
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt))
            total = 0.0
            for i in range(0, len(Xt), batch):
                idx = perm[i:i + batch]
                opt.zero_grad()
                loss = self.loss_fn_(self.model_(Xt[idx]).squeeze(1), yt[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.model_.eval()
            with torch.no_grad():
                val_loss = self.loss_fn_(self.model_(Xvt).squeeze(1), yvt).item()
            self.history_["train"].append(total / len(Xt))
            self.history_["val"].append(val_loss)
            if val_loss < best - 1e-5:
                best, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.best_epoch_ = int(np.argmin(self.history_["val"])) + 1
        return self

    def _raw(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(torch.from_numpy(np.asarray(X, dtype=np.float32))).squeeze(1).numpy()


class MLPRegressor(RegressorMixin, _TorchMLP):
    loss_fn_ = nn.MSELoss()

    def _prepare_targets(self, y):
        # Standardise the target for stable training; undone in predict().
        return (y - self.y_mean_) / self.y_std_

    def fit(self, X, y, eval_set=None):
        y = np.asarray(y, dtype=np.float32)
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
        return super().fit(X, y, eval_set)

    def predict(self, X):
        return self._raw(X) * self.y_std_ + self.y_mean_


class MLPClassifier(ClassifierMixin, _TorchMLP):
    loss_fn_ = nn.BCEWithLogitsLoss()

    def _prepare_targets(self, y):
        return y

    def fit(self, X, y, eval_set=None):
        self.classes_ = np.array([0, 1])
        return super().fit(X, y, eval_set)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self._raw(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
