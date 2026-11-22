"""Plots: model-selection curves and sensor readings with detected anomaly regions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _shade(ax, mask, color, alpha, label):
    """Shade contiguous True regions of `mask`."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for i, (s, e) in enumerate(zip(np.where(edges == 1)[0], np.where(edges == -1)[0])):
        ax.axvspan(s, e, color=color, alpha=alpha, label=label if i == 0 else None, lw=0)


def plot_detection(df, features, flags, starts, scores, threshold, window_size, path):
    fig, axes = plt.subplots(len(features) + 1, 1, figsize=(14, 3 * (len(features) + 1)), sharex=True)
    truth = df["is_anomaly"].to_numpy().astype(bool)
    for ax, feat in zip(axes, features):
        ax.plot(df[feat], lw=0.7, color="tab:blue")
        _shade(ax, truth, "gold", 0.35, "True anomaly")
        _shade(ax, flags, "red", 0.20, "Detected anomaly")
        ax.set_ylabel(feat)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Sensor readings with detected anomaly regions")

    ax = axes[-1]
    ax.plot(starts + window_size - 1, scores, lw=0.9, color="tab:green")
    ax.axhline(threshold, color="red", ls="--", label=f"Threshold = {threshold:.2f}")
    ax.set_ylabel("Window log-lik\n(per sample)")
    ax.set_xlabel("Time step")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_selection(table, chosen, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(table.n_states, table.bic, "o-"); a1.set_title("BIC (lower is better)")
    a2.plot(table.n_states, table.val_ll_per_sample, "o-", label="validation")
    a2.plot(table.n_states, table.train_ll_per_sample, "s--", label="train")
    a2.set_title("Log-likelihood per sample"); a2.legend()
    for ax in (a1, a2):
        ax.axvline(chosen, color="red", ls=":"); ax.set_xlabel("Hidden states")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
