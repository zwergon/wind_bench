import matplotlib.pyplot as plt
import numpy as np

from wbvirtual.train.predictions import Predictions
from wbvirtual.dataset import WBDataset


def smooth(liste, beta=0.98):
    avg = 0.0
    threshold = 0.0
    smoothed_list = []
    for i, l in enumerate(liste):
        # Compute the smoothed loss
        avg = beta * avg + (1 - beta) * l
        smoothed = avg / (1 - beta ** (i + 1))
        # Stop if the loss is exploding
        if i > len(liste) // 2 and smoothed >= threshold:
            break
        # Record the best loss
        if i == len(liste) // 3:
            threshold = smoothed
        smoothed_list.append(smoothed)
    return smoothed_list


def lrfind_plot(lr, loss):
    fig, ax = plt.subplots(figsize=(10, 5))
    trace = smooth(loss)
    ax.plot(lr[: len(loss)], loss, color="lightsteelblue", alpha=0.4)
    ax.plot(lr[: len(trace)], trace, color="navy")

    ax.set_title("LR Finder", fontsize=18)
    ax.set_xlabel("learning rate", fontsize=15)
    ax.set_ylabel("Loss", fontsize=15)
    ax.set_xscale("log")
    ax.set_xticks(
        np.array([np.arange(1, 10) * 10 ** (-8 + i) for i in range(1, 10)]).flatten()
    )
    ax.set_ylim(0.95 * min(trace), 1.05 * max(trace))

    return fig


def predictions_plot(predictions: Predictions, index=0, span=None):
    if span is not None:
        deb = span[0]
        end = span[1]
    else:
        deb = 0
        end = predictions.actual.shape[2]

    dataset: WBDataset = predictions.loader.dataset

    x_range = np.arange(deb, end, 1.0)

    fig, axs = plt.subplots(dataset.output_size, sharex=True, squeeze=False)
    fig.suptitle("Actual vs. Predicted")
    for i in range(dataset.output_size):
        y = predictions.actual[index, i, deb:end]
        y_hat = predictions.predicted[index, i, deb:end]

        axs[i][0].plot(x_range, y, label="Actual")
        axs[i][0].plot(x_range, y_hat, label="Predicted")
        axs[i][0].set_ylabel(dataset.output_name(i))
        axs[i][0].legend()

    return fig
