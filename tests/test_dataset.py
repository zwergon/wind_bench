import unittest
import os

from wbvirtual.dataset.dataloader import NaiveDataLoader
from wbvirtual.utils.time_utils import Timer
from wbvirtual.dataset import FileWBDataset, NumpyWBDataset


class TestDataset(unittest.TestCase):
    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.wb_root_path = os.path.join(os.path.dirname(__file__), "data/wb/100_128")
        self.toy_root_path = os.path.join(
            os.path.dirname(__file__), "data/fivestorybuilding/100_200"
        )

    def test_windbench_dataset(self):
        print("Wind Bench Dataset")

        filename = os.path.join(self.wb_root_path, "wind_bench.parquet")
        dataset = FileWBDataset(filename)

        print(f"train dataset (size) {len(dataset)}")
        self.assertEqual(len(dataset), 80)
        keys = dataset.partition_keys
        print(f"keys[0]: {keys[0]}")
        # self.assertEqual(keys[0], "Exp1.6")
        self.assertEqual(len(keys), 80)

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(
            f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)"
        )

    def test_toy_dataset(self):
        print("Toy Dataset")
        filename = os.path.join(self.toy_root_path, "fivestorybuilding.parquet")
        dataset = FileWBDataset(filename, sensor_desc="toy")

        print(f"train dataset (size) {len(dataset)}")
        self.assertEqual(len(dataset), 80)
        keys = dataset.partition_keys
        print(f"keys[0]: {keys[0]}")
        # self.assertEqual(keys[0], "Exp99")
        self.assertEqual(len(keys), 80)

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
        dataset = NumpyWBDataset(self.wb_root_path, indices=[0, 3])

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
