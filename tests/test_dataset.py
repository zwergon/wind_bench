import unittest
import os
import torch
from wind_bench.dataloader import NaiveDataLoader

from wind_bench.utils.time_utils import Timer
from wind_bench.dataset import S3WBDataset, FileWBDataset, NumpyWBDataset



class TestDataset(unittest.TestCase):

    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.root_path = os.path.join(os.path.dirname(__file__), "../../data/wb/100_1024")
        
    def test_file_dataset(self):
        
        filename = os.path.join(self.root_path, "wind_bench.parquet")
        dataset = FileWBDataset(filename)

        print(f"train dataset (size) {len(dataset)}")
        
        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")

    def test_s3_dataset(self):
        dataset = S3WBDataset("wind_bench.parquet")
        
        print(f"train dataset (size) {len(dataset)}")

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")

    def test_numpy_dataset(self):
        dataset = NumpyWBDataset(self.root_path, indices=[0, 3])
        
        print(f"train dataset (size) {len(dataset)}")

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")


if __name__ == "__main__":
    unittest.main()