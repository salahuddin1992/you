"""Classic machine learning from scratch (NumPy), complementing the neural
brain: k-means, k-nearest-neighbours, linear and logistic regression, and PCA.
Small, readable, and useful as reasoning tools on tabular data."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------- #
def kmeans(X: np.ndarray, k: int, iters: int = 100, seed: int = 0,
           tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm with k-means++ initialisation.
    Returns (centers [k,d], labels [n])."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        centers.append(X[rng.choice(n, p=probs)])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        new_labels = dist.argmin(1)
        new_centers = np.array([X[new_labels == j].mean(0) if (new_labels == j).any()
                                else centers[j] for j in range(k)])
        shift = np.abs(new_centers - centers).max()
        centers, labels = new_centers, new_labels
        if shift < tol:
            break
    return centers, labels


def knn_classify(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray,
                 k: int = 3) -> np.ndarray:
    """k-nearest-neighbours majority-vote classifier."""
    X_train = np.asarray(X_train, float); y_train = np.asarray(y_train)
    X_query = np.asarray(X_query, float)
    out = np.empty(len(X_query), dtype=y_train.dtype)
    for i, q in enumerate(X_query):
        d = ((X_train - q) ** 2).sum(1)
        idx = np.argsort(d)[:k]
        vals, counts = np.unique(y_train[idx], return_counts=True)
        out[i] = vals[counts.argmax()]
    return out


class LinearRegression:
    """Ordinary least squares via the normal equation (ridge-regularised)."""

    def __init__(self, l2: float = 0.0) -> None:
        self.l2 = l2
        self.w: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.hstack([np.ones((len(X), 1)), X])     # bias column
        A = Xb.T @ Xb + self.l2 * np.eye(Xb.shape[1])
        self.w = np.linalg.solve(A, Xb.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, float)])
        return Xb @ self.w


class LogisticRegression:
    """Binary logistic regression trained by gradient descent."""

    def __init__(self, lr: float = 0.1, epochs: int = 500, l2: float = 0.0) -> None:
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.w: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        X = np.asarray(X, float); y = np.asarray(y, float)
        Xb = np.hstack([np.ones((len(X), 1)), X])
        self.w = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-Xb @ self.w))
            grad = Xb.T @ (p - y) / len(y) + self.l2 * self.w
            self.w -= self.lr * grad
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xb = np.hstack([np.ones((len(X), 1)), np.asarray(X, float)])
        return 1.0 / (1.0 + np.exp(-Xb @ self.w))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


def pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Principal component analysis via SVD. Returns (projected, components)."""
    X = np.asarray(X, float)
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    return Xc @ comps.T, comps
