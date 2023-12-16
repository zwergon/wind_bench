import unittest
import os

from wb.dataset.dataloader import NaiveDataLoader
from wb.utils.time_utils import Timer
from wb.dataset import FileWBDataset, NumpyWBDataset


class TestDataset(unittest.TestCase):
    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.root_path = os.path.join(os.path.dirname(__file__), "data/100_128")

    def test_file_dataset(self):
        print("FileWBDataset")

        filename = os.path.join(self.root_path, "wind_bench.parquet")
        dataset = FileWBDataset(filename)

        print(f"train dataset (size) {len(dataset)}")
        print(dataset.partition_keys)

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(
            f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)"
        )

    def test_numpy_dataset(self):
        print("NumpyWBDataset")
        dataset = NumpyWBDataset(self.root_path, indices=[0, 3])

        print(f"train dataset (size) {len(dataset)}")

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(
            f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)"
        )


if __name__ == "__main__":
    unittest.main()
