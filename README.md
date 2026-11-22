# HMM Anomaly Detector for Machine Sensor Data

Trains a Gaussian-emission Hidden Markov Model on normal two-channel sensor data (temperature, vibration), then scans a continuous stream with a sliding window and flags windows whose likelihood is unusually low.

## Run
```bash
pip install -r requirements.txt
python run.py          # generates data (if missing), selects model, detects, plots
pytest
```
Outputs go to `outputs/`: `anomaly_detection.png`, `model_selection.png`, `model_selection.csv`, `metrics.json`.
To use your own data, place `train_normal.csv` and `test_stream.csv` (columns `temperature`, `vibration`, plus `is_anomaly` in the test file) in `data/`.

## Structure
| File | Purpose |
|---|---|
| `hmm_anomaly/data.py` | Simulated sensor generator (3 operating regimes; 3 injected fault types in test) |
| `hmm_anomaly/model.py` | HMM fitting with restarts, BIC model selection, window scoring, threshold |
| `hmm_anomaly/detector.py` | `StreamingDetector`: processes one sample at a time with a ring buffer |
| `hmm_anomaly/visualize.py` | Detection overlay and model-selection plots |
| `run.py` | End-to-end pipeline and evaluation |

## Method
1. **Scaling**: standardise with statistics from normal data only.
2. **Model selection**: fit 1-6 states on the first 75% of normal data (chronological split) and compare **BIC** and **held-out log-likelihood** on the last 25%. Each candidate keeps the best of 5 random EM restarts because EM is initialisation-sensitive.
3. **Threshold**: score all sliding windows of normal data (length 40, stride 5); the threshold is the **1st percentile** of per-sample window log-likelihood (-0.27). Scores are normalised by window length so the threshold does not depend on window size.
4. **Streaming**: `StreamingDetector.update(sample)` buffers the latest 40 samples and scores a window every 5 samples. A window below the threshold is flagged.
5. **Stationary start distribution**: windows begin at arbitrary points in a continuous stream, so the learned start probabilities (which describe only the first sample of training) are replaced by the chain's stationary distribution. Otherwise windows starting in a rarely-first regime would be penalised.

## Model selection: why 3 states
| States | BIC | Val LL/sample |
|---|---|---|
| 1 | 14461 | -1.62 |
| 2 | 4759 | -0.35 |
| **3** | **-6658** | **0.69** |
| 4 | -6577 | 0.69 |
| 5 | -6464 | 0.69 |
| 6 | -6348 | 0.69 |

BIC drops sharply up to 3 states and then rises, and validation likelihood plateaus at 3. More states only add parameters (overfitting) without improving fit on unseen data. 3 states also matches a physical interpretation: idle, normal load, high load. On real data the curve will be less clean, so prefer the smallest state count where validation likelihood stops improving.

## Results (simulated test stream, 3 injected faults)
- **Events detected: 3/3** (spike burst, gradual overheating drift, vibration variance increase)
- **False-alarm windows: 0** out of 593
- Point-wise precision 0.54, recall 1.00, F1 0.70.

Point-wise precision is low by construction: a flagged 40-sample window marks all its samples, so each detection extends about 30 samples beyond the true fault edges (visible as red margins in the plot). Event-level results are the better measure here. A smaller window or attributing the flag only to the newest samples would sharpen boundaries at the cost of noisier scores.

## Limitations
- The threshold percentile fixes the expected false-alarm rate on normal data (about 1% of windows). Choose it based on the cost of false alarms vs missed faults.
- Very slow drifts could be absorbed by the model's wide regimes; periodic retraining on recent normal data helps.
- Data is simulated; real sensors may need more states or non-Gaussian emissions.
