from torchmetrics import R2Score, ExplainedVariance


class MetricsCollection:
    def __init__(self, num_outputs, device) -> None:
        self.metrics = {
            "r2": R2Score(num_outputs=num_outputs, multioutput="raw_values").to(device),
            "explained_variance": ExplainedVariance(multioutput="raw_values"),
        }
        self.results = {k: None for k in self.metrics.keys()}

    def update_from_batch(self, Y_hat, Y):
        for b in range(Y.shape[0]):
            y_transposed = Y[b, :, :].transpose(0, 1)
            y_hat_transposed = Y_hat[b, :, :].transpose(0, 1)
            for m in self.metrics.values():
                m.update(y_hat_transposed, y_transposed)

    def compute(self):
        self.results = {k: m.compute() for k, m in self.metrics.items()}
        return self.results
