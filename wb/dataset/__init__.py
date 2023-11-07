import os
import glob
import torch
import numpy as np

from torch.utils.data import Dataset
import pyarrow.parquet as pq

from wb.dataset.s3 import S3
from wb.utils.time_utils import Timer



class WBDataset(Dataset):
    x_columns = [
            "Tower top fore-aft acceleration ay [m/s2]",
            "Tower top side-side acceleration ax [m/s2]",
            "Tower mid fore-aft acceleration ay [m/s2]",
            "Tower mid side-side acceleration ax [m/s2]",
            "Tower top rotation x [deg]",
            "Tower top rotation y [deg]",
            "Tower mid rotation x [deg]",
            "Tower mid rotation y [deg]"
        ]
    
    y_columns = [
        "Mudline moment Mx[kNm]",
        "Mudline moment My[kNm]",
        "Mudline moment Mz[kNm]",
        "Waterline moment Mx[kNm]",
        "Waterline moment My[kNm]",
        "Waterline moment Mz[kNm]"
    ]
    
    def __init__(self, parquet_file, train_flag, train_test_ratio):
       
        self.train_flag = train_flag
        self.train_test_ratio = train_test_ratio
        self.train = []
        self.test = []
        self.keys = []

    def __len__(self):
        return len(self.keys)
    
    def _split_train_test(self, keys: list):
        for _ in range(int(self.train_test_ratio*len(keys))):
            self.train.append(keys.pop())
        self.test = keys

        if self.train_flag:
            self.keys = self.train
        else:
            self.keys = self.test


class S3WBDataset(WBDataset):

    def __init__(self, parquet_file, train_flag=True, train_test_ratio=.8):
        super(S3WBDataset, self).__init__(parquet_file, train_flag=train_flag, train_test_ratio=train_test_ratio)
    
        keys = []
        with S3() as s3:
            for obj in s3.bucket.objects.filter(Prefix=parquet_file):
                keys.append(obj.key)

        self._split_train_test(keys)

    
    def __getitem__(self, idx):
        
        key = self.keys[idx]
        with S3() as s3:
            table = pq.read_table(
                f"{s3.bucket.name}/{key}", 
                filesystem=s3.filesystem, 
                columns=self.x_columns + self.y_columns
                )

        X = np.array(table.select(self.x_columns), dtype=np.float32)
        y = np.array(table.select(self.y_columns), dtype=np.float32)
        return X, y
    

class FileWBDataset(WBDataset):

    def __init__(self, parquet_file, train_flag=True, train_test_ratio=.8):
        super(FileWBDataset, self).__init__(parquet_file, train_flag=train_flag, train_test_ratio=train_test_ratio)

        keys = []
        for file in glob.glob("*/*.parquet", root_dir=parquet_file):
            keys.append(os.path.join(parquet_file, file))

        self._split_train_test(keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        table = pq.read_table(
            key, 
            columns=self.x_columns + self.y_columns
            )

        X = np.array(table.select(self.x_columns), dtype=np.float32)
        y = np.array(table.select(self.y_columns), dtype=np.float32)
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


from azureml.fsspec import AzureMachineLearningFileSystem


class AzureDataset(Dataset):
    x_columns = [
            "Tower top fore-aft acceleration ay [m/s2]",
            "Tower top side-side acceleration ax [m/s2]",
            "Tower mid fore-aft acceleration ay [m/s2]",
            "Tower mid side-side acceleration ax [m/s2]",
            "Tower top rotation x [deg]",
            "Tower top rotation y [deg]",
            "Tower mid rotation x [deg]",
            "Tower mid rotation y [deg]"
        ]
    
    y_columns = [
        "Mudline moment Mx[kNm]",
        "Mudline moment My[kNm]",
        "Mudline moment Mz[kNm]",
        "Waterline moment Mx[kNm]",
        "Waterline moment My[kNm]",
        "Waterline moment Mz[kNm]"
    ]
    
    def __init__(self, uri, train_flag, train_test_ratio):

        self.fs = AzureMachineLearningFileSystem(uri)
        self.train_flag = train_flag
        self.train_test_ratio = train_test_ratio
        self.train = []
        self.test = []
        self.keys = []

        keys = self.fs.glob("**/*.parquet")
        
        self._split_train_test(keys)

    @property
    def input_size(self):
        return len(self.x_columns)
    
    @property
    def output_size(self):
        return len(self.y_columns)

    def __getitem__(self, idx):
        key = self.keys[idx]
        table = pq.read_table(
            key, 
            columns=self.x_columns + self.y_columns,
            filesystem=self.fs
            )

        X = np.array(table.select(self.x_columns), dtype=np.float32)
        y = np.array(table.select(self.y_columns), dtype=np.float32)
        return X, y

    def __len__(self):
        return len(self.keys)
    
    def _split_train_test(self, keys: list):
        for _ in range(int(self.train_test_ratio*len(keys))):
            self.train.append(keys.pop())
        self.test = keys

        if self.train_flag:
            self.keys = self.train
        else:
            self.keys = self.test
