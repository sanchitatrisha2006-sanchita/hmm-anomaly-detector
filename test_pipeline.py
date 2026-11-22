import numpy as np
from sklearn.preprocessing import StandardScaler

from hmm_anomaly import config, data, model as hmm
from hmm_anomaly.detector import StreamingDetector, windows_to_point_flags


def _fitted():
    train = data.make_train(2000)[config.FEATURES].to_numpy()
    scaler = StandardScaler().fit(train)
    X = scaler.transform(train)
    m = hmm.use_stationary_start(hmm.fit_hmm(X, 3, n_restarts=2))
    return m, scaler, X


def test_start_distribution_is_valid():
    m, _, _ = _fitted()
    assert np.isclose(m.startprob_.sum(), 1) and (m.startprob_ >= 0).all()


def test_anomalous_window_scores_lower_than_normal():
    m, _, X = _fitted()
    _, scores = hmm.window_scores(m, X, config.WINDOW_SIZE, config.STRIDE)
    anomalous = X[:config.WINDOW_SIZE] + 6.0
    assert m.score(anomalous) / config.WINDOW_SIZE < np.percentile(scores, 1)


def test_streaming_emits_one_result_per_stride():
    m, scaler, _ = _fitted()
    det = StreamingDetector(m, scaler, -1.0, window_size=40, stride=5)
    test = data.make_test()[config.FEATURES].iloc[:200].to_numpy()
    results = [r for row in test if (r := det.update(row)) is not None]
    assert len(results) == (200 - 40) // 5 + 1
    assert windows_to_point_flags(results, 200, 40).shape == (200,)
