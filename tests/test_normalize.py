import unittest
import os
import json

import numpy as np

from wb.utils.time_utils import Timer
from wb.dataset import FileWBDataset
from wb.dataset.normalize import Normalize, MinMax

import matplotlib.pyplot as plt



class TestNormalize(unittest.TestCase):

    batch_size = 8

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.root_path = os.path.join(os.path.dirname(__file__), "data")
        with open(os.path.join(self.root_path, "stats.json"), 'r') as f:
            self.stats = json.load(f)
        self.idx = 0
        
    def _test_normalize(self, norma: Normalize):
        
        filename = os.path.join(self.root_path, "100_128", "wind_bench.parquet")
        
        dataset = FileWBDataset(filename, normalization=None)
        print(f"train dataset (size) {len(dataset)}")
           
        X, Y = dataset[0]
        print(np.mean(X[self.idx, :]), self.stats[FileWBDataset.x_columns[self.idx]]['mean'])
         
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

        norma = Normalize(self.stats, FileWBDataset.x_columns, FileWBDataset.y_columns)
        self._test_normalize(norma)

    def test_minmax(self):
        
        norma = MinMax(self.stats, FileWBDataset.x_columns, FileWBDataset.y_columns)
        self._test_normalize(norma)

       



if __name__ == "__main__":
    unittest.main()