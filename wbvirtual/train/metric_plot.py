import torch
import numpy as np
import matplotlib.pyplot as plt
from wbvirtual.train.predictions import Predictions

class MetricsBoxPlot:
    def __init__(self, metric):
        self.metric = metric

    def compute(self, predictions: Predictions):
        Y = predictions.actual
        y_hat = predictions.predicted

        result_list = [
            [self.metric(Y[i, j, :], y_hat[i, j, :]) for j in range(Y.shape[1])]
            for i in range(Y.shape[0])
        ]

        result_list = np.transpose(result_list)
        result_list = [np.array(sub_array) for sub_array in result_list]
        labels = [f"col_{i+1}" for i in range(len(result_list))]

        fig, ax = plt.subplots()
        ax.boxplot(result_list, vert=True, patch_artist=True, labels=labels)

        ax.set_title("Score")
        ax.yaxis.grid(True)
        ax.set_ylabel("Observed values")

        return fig
