"""HMM training, model selection and window scoring."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from hmm_anomaly import config


def fit_hmm(X: np.ndarray, n_states: int, n_restarts: int = config.N_RESTARTS) -> GaussianHMM:
    """Fit a Gaussian-emission HMM, keeping the best of several random restarts."""
    best, best_ll = None, -np.inf
    for i in range(n_restarts):
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=200,
                            tol=1e-3, random_state=config.RANDOM_STATE + i)
        try:
            model.fit(X)
            ll = model.score(X)
        except ValueError:  # degenerate fit (e.g. singular covariance)
            continue
        if ll > best_ll:
            best, best_ll = model, ll
    if best is None:
        raise RuntimeError(f"All {n_restarts} fits failed for n_states={n_states}")
    return best


def n_parameters(n_states: int, n_features: int) -> int:
    """Free parameters of a full-covariance Gaussian HMM."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covs = n_states * n_features * (n_features + 1) // 2
    return start + trans + means + covs


def select_n_states(X_train: np.ndarray, X_val: np.ndarray) -> pd.DataFrame:
    """Evaluate each candidate state count with BIC (train) and held-out log-likelihood (val)."""
    rows = []
    for k in config.CANDIDATE_STATES:
        model = fit_hmm(X_train, k)
        ll_train = model.score(X_train)
        bic = -2 * ll_train + n_parameters(k, X_train.shape[1]) * np.log(len(X_train))
        rows.append({"n_states": k, "bic": bic,
                     "train_ll_per_sample": ll_train / len(X_train),
                     "val_ll_per_sample": model.score(X_val) / len(X_val)})
    return pd.DataFrame(rows)


def use_stationary_start(model: GaussianHMM) -> GaussianHMM:
    """Replace the start distribution with the chain's stationary distribution.

    Windows in a continuous stream begin at arbitrary times, so the learned start
    distribution (which describes the very first sample only) would unfairly penalise them.
    """
    eigvals, eigvecs = np.linalg.eig(model.transmat_.T)
    stationary = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
    model.startprob_ = np.abs(stationary) / np.abs(stationary).sum()
    return model


def window_scores(model: GaussianHMM, X: np.ndarray, size: int, stride: int):
    """Length-normalised log-likelihood of each sliding window.

    Returns (starts, scores); score = log P(window | model) / size so thresholds are
    independent of window length.
    """
    starts = np.arange(0, len(X) - size + 1, stride)
    scores = np.array([model.score(X[s:s + size]) / size for s in starts])
    return starts, scores


@dataclass
class Threshold:
    value: float
    percentile: float


def fit_threshold(model, X_normal, size, stride, percentile=config.THRESHOLD_PERCENTILE) -> Threshold:
    _, scores = window_scores(model, X_normal, size, stride)
    return Threshold(float(np.percentile(scores, percentile)), percentile)
