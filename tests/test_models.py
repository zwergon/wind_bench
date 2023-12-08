import unittest
import os
import torch

from wb.dataset import dataset
from wb.virtual.models import get_model
from wb.utils.config import Config
from wb.virtual.context import Context


class TestModel(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.config = Config(os.path.join(os.path.dirname(__file__), "config.json"))
         
    def test_cnn(self):

        context = Context(self.config)

        train_dataset, _ = dataset(self.config)
        X, y = train_dataset[0]
        print(X.shape, y.shape)

        model =  get_model(context, train_dataset.input_size, train_dataset.output_size )
        print(model)


        X_tf = torch.from_numpy(X).view(1, X.shape[0], -1)
        print(X_tf.shape)
        predicted = model(X_tf)
        print(predicted.shape)


if __name__ == "__main__":
   unittest.main() 
