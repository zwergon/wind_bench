import unittest
import os
import torch

from wbvirtual.dataset import FileWBDataset
from wbvirtual.train.models import get_model, ModelError
from wbvirtual.utils.config import Config
from wbvirtual.train.context import Context


class TestModel(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.wb_root_path = os.path.join(os.path.dirname(__file__), "data/wb/100_128")
        self.config = Config.test_config()

    def test_cnn(self):
        context = Context(self.config)

        filename = os.path.join(self.wb_root_path, "wind_bench.parquet")
        train_dataset = FileWBDataset(filename)
        X, y = train_dataset[0]
        print(X.shape, y.shape)

        model = get_model(
            context.config,
            train_dataset.input_size,
            train_dataset.output_size,
            context.device,
        )
        print(model)

        X_tf = torch.from_numpy(X).view(1, X.shape[0], -1)
        print(X_tf.shape)
        predicted = model(X_tf)
        print(predicted.shape)

    def test_unknown_model(self):
        context = Context(self.config)
        context.config["type"] = "unknown"

        try:
            model = None
            model = get_model(
                context.config,
                8,
                1,
                context.device,
            )
        except ModelError:
            self.assertTrue(model is None)


if __name__ == "__main__":
    unittest.main()
