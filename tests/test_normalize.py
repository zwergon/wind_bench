import unittest
import os
import json

import numpy as np

from wb.dataset import FileWBDataset
from wb.dataset.normalize import Normalize, MinMax
from wb.dataset.sensor_description import WBSensorDescr

import matplotlib.pyplot as plt


class TestNormalize(unittest.TestCase):
    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.root_path = os.path.join(os.path.dirname(__file__), "data", "wb")
        self.idx = 0

    def _test_normalize(self, norma: Normalize):
        filename = os.path.join(self.root_path, "100_128", "wind_bench.parquet")

        dataset = FileWBDataset(filename, normalization=None)
        print(f"train dataset (size) {len(dataset)}")

        X, Y = dataset[0]
        print(
            np.mean(X[self.idx, :]),
            dataset.stats[dataset.x_columns[self.idx]]["mean"],
        )

        plt.plot(Y[0, :])
        plt.show()

        print(Y.shape)

        norma.norm_y(Y)

        plt.plot(Y[0, :])
        plt.show()

        norma.unnorm_y(Y)

        plt.plot(Y[0, :])
        plt.show()

    def test_normalize(self):
        sensor_descr = WBSensorDescr()
        norma = Normalize(
            sensor_descr.stats, sensor_descr.x_columns, sensor_descr.y_columns
        )
        self._test_normalize(norma)

    def test_minmax(self):
        sensor_descr = WBSensorDescr()
        norma = MinMax(
            sensor_descr.stats, sensor_descr.x_columns, sensor_descr.y_columns
        )
        self._test_normalize(norma)


if __name__ == "__main__":
    unittest.main()
