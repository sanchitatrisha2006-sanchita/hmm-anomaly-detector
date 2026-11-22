"""Simulated machine-sensor data (the assignment's 'provided' data is replaced by this generator).

Normal operation switches between three regimes (idle, normal load, high load) following a
Markov chain. Test data adds three kinds of faults: a spike burst, a gradual drift and a
vibration-variance increase (e.g. a failing bearing).
"""
import numpy as np
import pandas as pd

from hmm_anomaly import config

# (mean temperature, mean vibration, std temperature, std vibration) per regime
_REGIMES = np.array([
    [40.0, 0.5, 1.0, 0.10],   # idle
    [60.0, 1.5, 1.5, 0.20],   # normal load
    [80.0, 2.5, 2.0, 0.30],   # high load
])
_TRANSITIONS = np.array([
    [0.97, 0.03, 0.00],
    [0.02, 0.96, 0.02],
    [0.00, 0.04, 0.96],
])


def _simulate_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    states = np.empty(n, dtype=int)
    states[0] = rng.integers(len(_REGIMES))
    for t in range(1, n):
        states[t] = rng.choice(len(_REGIMES), p=_TRANSITIONS[states[t - 1]])
    means, stds = _REGIMES[states, :2], _REGIMES[states, 2:]
    return means + stds * rng.standard_normal((n, 2))


def make_train(n: int = 6000, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    return pd.DataFrame(_simulate_normal(n, np.random.default_rng(seed)), columns=config.FEATURES)


def make_test(n: int = 3000, seed: int = config.RANDOM_STATE + 1) -> pd.DataFrame:
    """Normal stream with injected anomalies; `is_anomaly` holds the ground truth."""
    rng = np.random.default_rng(seed)
    x = _simulate_normal(n, rng)
    label = np.zeros(n, dtype=int)

    def mark(start, end):
        label[start:end] = 1

    # 1) spike burst: sudden temperature + vibration excursions
    s, e = 600, 640
    x[s:e] += rng.normal([25, 2.0], [6, 0.8], size=(e - s, 2)); mark(s, e)
    # 2) gradual temperature drift (overheating)
    s, e = 1500, 1600
    x[s:e, 0] += np.linspace(0, 30, e - s); mark(s, e)
    # 3) vibration variance increase (bearing wear)
    s, e = 2300, 2380
    x[s:e, 1] += rng.normal(0, 0.9, e - s); mark(s, e)

    df = pd.DataFrame(x, columns=config.FEATURES)
    df["is_anomaly"] = label
    return df


def save_datasets() -> None:
    config.DATA_DIR.mkdir(exist_ok=True)
    make_train().to_csv(config.TRAIN_PATH, index=False)
    make_test().to_csv(config.TEST_PATH, index=False)


if __name__ == "__main__":
    save_datasets()
    print(f"Wrote {config.TRAIN_PATH} and {config.TEST_PATH}")
