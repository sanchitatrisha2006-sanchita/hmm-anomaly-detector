"""Real-time sliding-window anomaly detector."""
from collections import deque
from dataclasses import dataclass

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


@dataclass
class WindowResult:
    end_index: int      # index of the newest sample in the window
    score: float        # per-sample log-likelihood
    is_anomaly: bool


class StreamingDetector:
    """Feed samples one at a time with `update`; a window is scored every `stride` samples."""

    def __init__(self, model: GaussianHMM, scaler: StandardScaler, threshold: float,
                 window_size: int, stride: int):
        self.model, self.scaler = model, scaler
        self.threshold, self.window_size, self.stride = threshold, window_size, stride
        self._buffer = deque(maxlen=window_size)
        self._count = 0

    def update(self, sample) -> WindowResult | None:
        """Add one raw sample; return a result when a window has just been scored."""
        self._buffer.append(self.scaler.transform(np.asarray(sample, dtype=float).reshape(1, -1))[0])
        self._count += 1
        full = len(self._buffer) == self.window_size
        if not full or (self._count - self.window_size) % self.stride != 0:
            return None
        score = self.model.score(np.array(self._buffer)) / self.window_size
        return WindowResult(self._count - 1, float(score), bool(score < self.threshold))


def windows_to_point_flags(results: list[WindowResult], n_samples: int, window_size: int) -> np.ndarray:
    """Mark every sample covered by at least one anomalous window."""
    flags = np.zeros(n_samples, dtype=bool)
    for r in results:
        if r.is_anomaly:
            flags[r.end_index - window_size + 1: r.end_index + 1] = True
    return flags
