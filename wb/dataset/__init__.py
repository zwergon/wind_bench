import os
import re
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from wb.dataset.normalize import Scaling
from wb.utils.config import Config, FSType, FileType
from wb.dataset.sensor_description import SensorDescr


class WBDataset(Dataset):
    _pattern = re.compile(r"[\/|\\]\w+=([\w|\.]+)[\/|\\]")

    def __init__(
        self, train_flag, train_test_ratio, sensor_desc, normalization, indices, seed
    ):
        self.sensor_desc = SensorDescr.create(sensor_desc)
        self.train_flag = train_flag
        self.train_test_ratio = train_test_ratio
        self.train = []
        self.test = []
        self.keys = []
        random.seed(seed)

        if indices is None:
            self.y_selected = self.y_columns
        else:
            self.y_selected = [self.y_columns[i] for i in indices]

        self.norma: Scaling = Scaling.create(
            normalization, self.stats, self.x_columns, self.y_selected
        )

    def __len__(self):
        return len(self.keys)

    @property
    def input_size(self):
        return len(self.x_columns)

    @property
    def stats(self):
        return self.sensor_desc.stats

    @property
    def x_columns(self):
        return self.sensor_desc.x_columns

    @property
    def y_columns(self):
        return self.sensor_desc.y_columns

    @property
    def partition_keys(self):
        keys = []
        for k in self.keys:
            m = re.search(self._pattern, k)
            if m:
                keys.append(m.group(1))
        return keys

    @property
    def output_size(self):
        return len(self.y_selected)

    def output_name(self, index):
        return self.y_selected[index]

    def _split_train_test(self, keys: list):
        indices = list(range(int(self.train_test_ratio * len(keys))))
        random.shuffle(indices)
        for _ in indices:
            self.train.append(keys.pop())
        self.test = keys

        if self.train_flag:
            self.keys = self.train
        else:
            self.keys = self.test


class FileWBDataset(WBDataset):
    @staticmethod
    def _search_parquets(rootdir, file_list):
        for f in os.listdir(rootdir):
            path = os.path.join(rootdir, f)
            if os.path.isfile(path) and path.endswith(".parquet"):
                file_list.append(path)
            else:
                FileWBDataset._search_parquets(path, file_list)

    def __init__(
        self,
        parquet_file,
        train_flag=True,
        train_test_ratio=0.8,
        sensor_desc="wind_bench",
        normalization="min_max",
        indices=None,
        seed=12,
    ):
        super(FileWBDataset, self).__init__(
            train_flag=train_flag,
            train_test_ratio=train_test_ratio,
            sensor_desc=sensor_desc,
            normalization=normalization,
            indices=indices,
            seed=seed,
        )

        keys = []
        self._search_parquets(parquet_file, keys)

        self._split_train_test(keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        df = pd.read_parquet(key, columns=self.x_columns + self.y_selected)

        X = df.loc[:, self.x_columns].transpose().to_numpy(dtype=np.float32)
        y = df.loc[:, self.y_selected].transpose().to_numpy(dtype=np.float32)
        if self.norma is not None:
            self.norma.norm_x(X)
            self.norma.norm_y(y)
        return X, y


class NumpyWBDataset(Dataset):
    def __init__(self, root_path, train_flag=True, indices=None):
        name = "train" if train_flag else "test"

        x_name = os.path.join(root_path, f"{name}_data.npy")
        y_name = os.path.join(root_path, f"{name}_labels.npy")
        self.X = np.load(x_name).astype(np.float32)
        self.y = np.load(y_name).astype(np.float32)

        if indices is None:
            self.indices = list(range(self.y.shape[1]))
        else:
            self.indices = indices

    def __len__(self):
        return self.X.shape[0]

    @property
    def input_size(self):
        return self.X.shape[1]

    @property
    def output_size(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.X[idx, :, :], self.y[idx, self.indices, :]
