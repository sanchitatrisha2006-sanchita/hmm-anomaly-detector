"""End-to-end pipeline: select model -> train -> threshold -> stream test data -> evaluate -> plot.

Usage: python run.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from hmm_anomaly import config, data, model as hmm, visualize
from hmm_anomaly.detector import StreamingDetector, windows_to_point_flags


def main():
    if not config.TRAIN_PATH.exists():
        data.save_datasets()
    train = pd.read_csv(config.TRAIN_PATH)
    test = pd.read_csv(config.TEST_PATH)
    config.OUTPUT_DIR.mkdir(exist_ok=True)

    scaler = StandardScaler().fit(train[config.FEATURES].to_numpy())
    X = scaler.transform(train[config.FEATURES].to_numpy())
    split = int(len(X) * (1 - config.VALIDATION_FRACTION))   # chronological split, no shuffling
    X_fit, X_val = X[:split], X[split:]

    # 1) model selection
    table = hmm.select_n_states(X_fit, X_val)
    chosen = int(table.loc[table.bic.idxmin(), "n_states"])
    table.to_csv(config.OUTPUT_DIR / "model_selection.csv", index=False)
    visualize.plot_model_selection(table, chosen, config.OUTPUT_DIR / "model_selection.png")
    print(table.round(3).to_string(index=False), f"\nChosen n_states = {chosen}\n")

    # 2) final model on all normal data, threshold from held-out-style window scores
    final = hmm.use_stationary_start(hmm.fit_hmm(X, chosen))
    threshold = hmm.fit_threshold(final, X, config.WINDOW_SIZE, config.STRIDE)
    print(f"Threshold (p{threshold.percentile}) = {threshold.value:.3f}")

    # 3) simulate real-time streaming over the test data
    detector = StreamingDetector(final, scaler, threshold.value, config.WINDOW_SIZE, config.STRIDE)
    results = [r for row in test[config.FEATURES].to_numpy() if (r := detector.update(row)) is not None]
    flags = windows_to_point_flags(results, len(test), config.WINDOW_SIZE)

    # 4) evaluate (point-wise) and plot
    truth = test["is_anomaly"].to_numpy()
    p, r, f1, _ = precision_recall_fscore_support(truth, flags, average="binary", zero_division=0)
    # event-level view: a window smears detections over `window_size` samples, so point-wise
    # precision understates quality. Count true events hit and false-alarm windows instead.
    events = np.flatnonzero(np.diff(np.r_[0, truth, 0]) != 0).reshape(-1, 2)
    events_detected = int(sum(flags[s:e].any() for s, e in events))
    false_alarm_windows = int(sum(x.is_anomaly and not truth[x.end_index - config.WINDOW_SIZE + 1: x.end_index + 1].any()
                                  for x in results))
    metrics = {"n_states": chosen, "threshold": threshold.value,
               "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
               "events_detected": f"{events_detected}/{len(events)}",
               "false_alarm_windows": false_alarm_windows, "windows_scored": len(results), "windows_flagged": sum(x.is_anomaly for x in results)}
    (config.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    starts = np.array([x.end_index - config.WINDOW_SIZE + 1 for x in results])
    scores = np.array([x.score for x in results])
    visualize.plot_detection(test, config.FEATURES, flags, starts, scores, threshold.value,
                             config.WINDOW_SIZE, config.OUTPUT_DIR / "anomaly_detection.png")


if __name__ == "__main__":
    main()
