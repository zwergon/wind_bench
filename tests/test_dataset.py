import unittest
import os
import torch

from wb.dataset.dataloader import NaiveDataLoader
from wb.utils.time_utils import Timer
from wb.dataset import S3WBDataset, FileWBDataset, NumpyWBDataset, AzureBlobDataset, AzureMLDataset


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
        
        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")
        

    # def test_s3_dataset(self):
    #     print("S3WBDataset")
    #     dataset = S3WBDataset("wind_bench.parquet")
        
    #     print(f"train dataset (size) {len(dataset)}")

    #     dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
    #     with Timer() as timer:
    #         for batch in dataloader:
    #             X, y = batch
    #             print(X.shape, y.shape)
    #             break

    #     print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")

    # def test_az_blob_dataset(self):
    #     print("AzureBlobDataset")
    #     dataset = AzureBlobDataset("wind_bench", indices=[0, 3])
        
    #     print(f"train dataset (size) {len(dataset)}")

    #     dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
    #     with Timer() as timer:
    #         for batch in dataloader:
    #             X, y = batch
    #             print(X.shape, y.shape)
    #             break

    #     dataset.close()

    #    print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")

    def test_az_ml_dataset(self):
        print("AzureMLDataset")
        dataset = AzureMLDataset("azureml://subscriptions/8a889cf2-7b3d-4003-824e-b503f56604b0/resourcegroups/rg-jef-ml/workspaces/ml-gpu/datastores/workspaceblobstore/paths/UI/2023-11-29_114659_UTC/wind_bench_1000_1028.parquet", indices=[0, 3])
        
        print(f"train dataset (size) {len(dataset)}")

        dataloader = NaiveDataLoader(dataset, batch_size=self.batch_size)
        with Timer() as timer:
            for batch in dataloader:
                X, y = batch
                print(X.shape, y.shape)
                break

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")

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

        print(f"load batch size {self.batch_size} in {float(timer):.2f}s ({self.batch_size/float(timer):.2f} i/s)")


if __name__ == "__main__":
    unittest.main()