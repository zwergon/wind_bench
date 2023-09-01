import unittest
import os
import torch

from wind_bench.dataset import S3WBDataset, FileWBDataset, NumpyWBDataset
from virtual.models import get_model
from args import Args

class TestDataset(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.args = Args(os.path.join(os.path.dirname(__file__), "args.json"))
        self.args.root_path = os.path.join(os.path.dirname(__file__), "data")
        
    def test_cnn(self):
        dataset = NumpyWBDataset(os.path.abspath(self.args.data_dir))
        X, y = dataset[0]
        print(X.shape, y.shape)

        

        model =  get_model(dataset.input_size, dataset.output_size, self.args.__dict__ )
        print(model)


        X_tf = torch.from_numpy(X).view(1, X.shape[0], -1)
        print(X_tf.shape)
        predicted = model(X_tf)
        print(predicted.shape)


if __name__ == "__main__":
   unittest.main() 
