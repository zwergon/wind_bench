import torch
import numpy as np
import matplotlib.pyplot as plt
from wbvirtual.train.predictions import Predictions


class MetricsBoxPlot:
    def __init__(self, metric):
        self.metric = metric

    def compute(self, predictions: Predictions):
        result_list = []
        labels = []
        Y = torch.tensor(predictions.actual)
        y_hat = torch.tensor(predictions.predicted)

        for d in range(Y.size(0)):
            trans_input = Y[d, :, :].transpose(0, 1)
            trans_target = y_hat[d, :, :].transpose(0, 1)
            results = self.metric(trans_input, trans_target)
            result_list.append(results.numpy())
        transposed_arrays = np.transpose(result_list)
        result_list = [np.array(sub_array) for sub_array in transposed_arrays]

        for i in range(len(result_list)):
            labels.append(f"col_{i+1}")
        fig, ax = plt.subplots()
        ax.boxplot(result_list, vert=True, patch_artist=True, labels=labels)

        ax.set_title("Score")
        ax.yaxis.grid(True)
        ax.set_ylabel("Observed values")

        return fig
