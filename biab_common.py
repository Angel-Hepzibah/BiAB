"""
biab_common.py — shared utilities for the Path #1 BiAB-IoT experiments.

All notebooks in this package import from here so that data loading, model
construction, federated averaging, and evaluation are identical across
baselines. Import via:

    from biab_common import (
        load_dataset, make_clients, make_model, train_local, fed_average,
        evaluate, poison_clients, save_result, load_all_results,
        RESULT_DIR, DATASET_PATH,
    )
"""

from __future__ import annotations
import os
import pickle
import random
import time
from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# Paths -- edit if you keep files elsewhere on Drive
DATASET_PATH = "/content/drive/MyDrive/binary_dataset.pkl"
RESULT_DIR = "/content/drive/MyDrive/path1_results"

# Global seed used everywhere; keep at 42 to match BIAB.ipynb
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_dataset(path: str = DATASET_PATH):
    """Load the binary_dataset.pkl produced by BIAB.ipynb.

    Returns a dict with keys X_train, X_test, y_train, y_test, client_data.
    y_train / y_test are pandas Series with the row order preserved from the
    train/test split (random_state=42 in BIAB.ipynb).
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def make_clients(X_train, y_train, num_clients: int = 500, seed: int = SEED):
    """IID partition of the training set into `num_clients` shards."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_train))
    size = len(idx) // num_clients
    clients = []
    for i in range(num_clients):
        s, e = i * size, (i + 1) * size
        clients.append((X_train[idx[s:e]], y_train.iloc[idx[s:e]]))
    return clients


def poison_clients(clients, pct: float, seed: int = SEED):
    """Label-flip a fraction `pct` of clients and return
    (poisoned_clients, malicious_ids)."""
    n = len(clients)
    n_mal = int(round(n * pct))
    rng = random.Random(seed)
    mal_ids = set(rng.sample(range(n), n_mal))
    out = []
    for cid, (Xc, yc) in enumerate(clients):
        if cid in mal_ids:
            out.append((Xc, 1 - yc))          # flip labels
        else:
            out.append((Xc, yc))
    return out, mal_ids


# ---------------------------------------------------------------------------
# Federated model + training
# ---------------------------------------------------------------------------

def make_model(input_dim: int):
    """The reference dense binary classifier used across all methods.

    Kept identical to BIAB.ipynb so that comparisons are apples-to-apples.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input
    m = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(64,  activation="relu"),
        Dense(1,   activation="sigmoid"),
    ])
    m.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m


def train_local(Xc, yc, global_weights, input_dim: int,
                epochs: int = 1, batch_size: int = 64):
    """Train one client for `epochs` epochs starting from `global_weights`.
    Returns the trained weights."""
    m = make_model(input_dim)
    m.set_weights(global_weights)
    m.fit(Xc, yc, epochs=epochs, batch_size=batch_size, verbose=0)
    return m.get_weights()


def fed_average(list_of_weights, weights_scale=None):
    """Weighted federated averaging.

    list_of_weights : list of client weight-lists (each a list of ndarrays).
    weights_scale   : optional list of scalar client weights (e.g. reputation).
                      If None, unweighted mean is used (== FedAvg).
    """
    if weights_scale is None:
        return [np.mean(w, axis=0) for w in zip(*list_of_weights)]
    ws = np.asarray(weights_scale, dtype=np.float64)
    ws = ws / ws.sum() if ws.sum() > 0 else np.ones_like(ws) / len(ws)
    out = []
    for tensors in zip(*list_of_weights):
        stacked = np.stack(tensors, axis=0)                          # (C, ...)
        # broadcast reputation weights over trailing dims
        shape = [len(ws)] + [1] * (stacked.ndim - 1)
        out.append(np.sum(stacked * ws.reshape(shape), axis=0))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test):
    """Return a metrics dict for a trained model on the held-out test set."""
    pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    y = np.asarray(y_test).astype(int).flatten()
    cm = confusion_matrix(y, pred)
    # sklearn convention: rows are actual, cols are predicted; label 0 first
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":  accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall":    recall_score(y, pred, zero_division=0),
        "f1":        f1_score(y, pred, zero_division=0),
        "fpr":       fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------------
# Result persistence
# ---------------------------------------------------------------------------

def save_result(name: str, metrics: dict, config: dict | None = None):
    """Persist metrics + config for later aggregation."""
    os.makedirs(RESULT_DIR, exist_ok=True)
    payload = {"name": name, "metrics": metrics,
               "config": config or {}, "timestamp": time.time()}
    path = os.path.join(RESULT_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved {path}")


def load_all_results(prefix: str | None = None):
    """Load every result pickle under RESULT_DIR (optionally filtered by prefix)."""
    out = []
    if not os.path.isdir(RESULT_DIR):
        return out
    for fn in sorted(os.listdir(RESULT_DIR)):
        if not fn.endswith(".pkl"):
            continue
        if prefix is not None and not fn.startswith(prefix):
            continue
        with open(os.path.join(RESULT_DIR, fn), "rb") as f:
            out.append(pickle.load(f))
    return out


# ---------------------------------------------------------------------------
# Small helper: set seeds everywhere
# ---------------------------------------------------------------------------

def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
