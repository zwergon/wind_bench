import torch
from wbvirtual.train.predictions import Predictions


class Metrics_boxplot:
    def __init__(self, metric):
        self.metric = metric

    def compute(self, predictions: Predictions):
        Y = torch.tensor(predictions.actual)
        y_hat = torch.tensor(predictions.predicted)
