import os
import glob
import torch
import numpy as np

from torch.utils.data import Dataset
import pyarrow.parquet as pq

from wb.dataset.s3 import S3
from wb.dataset.normalize import Scaling, MinMax
from wb.utils.time_utils import Timer



class WBDataset(Dataset):

    stats = {"Tower top fore-aft acceleration ay [m/s2]": { "mean": -3.732385818133257e-07, "std": 0.1175451058459768, "min": -0.8967088086, "25%": -0.07215977105999999, "50%": 7.251252000000028e-05, "75%": 0.07217759763, "max": 0.8616838365799999}, 
             "Tower top side-side acceleration ax [m/s2]": { "mean": -4.3043763994239984e-05, "std": 0.17042526932798652, "min": -1.7410713982200001, "25%": -0.08029902833, "50%": -5.848058999999989e-05, "75%": 0.08026554234000001, "max": 1.6791001381199997}, 
             "Tower mid fore-aft acceleration ay [m/s2]": { "mean": 7.059205544883323e-06, "std": 0.3120427307662817, "min": -3.1043631840000003, "25%": -0.1542743856, "50%": 0.0, "75%": 0.1542152164, "max": 2.8346083776}, 
             "Tower mid side-side acceleration ax [m/s2]": { "mean": -1.010584451681167e-05, "std": 0.20297381501506764, "min": -1.73153678728, "25%": -0.0980160628, "50%": -6.521140000000439e-06, "75%": 0.09805821861, "max": 1.7674581912}, 
             "Tower top rotation x [deg]": { "mean": 0.10906119975000214, "std": 0.22879160967145254, "min": -0.5287499999999454, "25%": -0.05624999999997726, "50%": 0.114750000000015, "75%": 0.2756250000000193, "max": 0.757125000000002}, 
             "Tower top rotation y [deg]": { "mean": -0.13664792117511612, "std": 0.06582604013556786, "min": -0.48334404635999995, "25%": -0.186245273584, "50%": -0.14719013135999998, "75%": -0.08372635442349999, "max": 0.02261905472}, 
             "Tower mid rotation x [deg]": { "mean": 0.20820803906250568, "std": 0.12007967185727536, "min": -0.12374999999997272, "25%": 0.12375000000000114, "50%": 0.20812499999999545, "75%": 0.29812499999999886, "max": 0.5343750000000114}, 
             "Tower mid rotation y [deg]": { "mean": -0.03926080885200301, "std": 0.02261261187435961, "min": -0.21044894401800002, "25%": -0.053850658272500004, "50%": -0.038773657971, "75%": -0.021825153801499998, "max": 0.05379317636400001}, 
             "Mudline moment Mx[kNm]": { "mean": 61977.27387588217, "std": 29619.62022923741, "min": -27907.36832, "25%": 40270.44517200001, "50%": 61874.379203000004, "75%": 83967.554778, "max": 152381.0179}, 
             "Mudline moment My[kNm]": { "mean": 7097.506250106164, "std": 5128.729624126434, "min": -26849.612592, "25%": 3270.19544245, "50%": 6418.2234465, "75%": 10052.65925425, "max": 53207.646748}, 
             "Mudline moment Mz[kNm]": { "mean": -711.7177369156096, "std": 6796.74697762426, "min": -49612.090249999994, "25%": -4573.1283393, "50%": -535.7095240000001, "75%": 3390.6702348500007, "max": 43376.417460000004}, 
             "Waterline moment Mx[kNm]": { "mean": 136798.9978638337, "std": 68450.42487856085, "min": -54995.24456400001, "25%": 87216.35613, "50%": 136273.3443, "75%": 187669.13632800002, "max": 330240.99749999994}, 
             "Waterline moment My[kNm]": { "mean": 18307.588403021604, "std": 11959.074576665971, "min": -49589.05464, "25%": 9239.241472999998, "50%": 17186.058381000003, "75%": 25477.165405999996, "max": 119527.84760000001}, 
             "Waterline moment Mz[kNm]": { "mean": -711.7178628248745, "std": 6789.644633459557, "min": -49547.412064, "max": 43301.68051}}
    
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
    
    def __init__(self, 
                 train_flag, 
                 train_test_ratio, 
                 normalization
                 ):
       
        self.train_flag = train_flag
        self.train_test_ratio = train_test_ratio
        self.train = []
        self.test = []
        self.keys = []
        
        self.norma : Scaling = Scaling.create(
            normalization,
            WBDataset.stats, 
            self.x_columns, 
            self.y_columns
            )
       
    def __len__(self):
        return len(self.keys)
    
    @property
    def input_size(self):
        return len(self.x_columns)

    @property
    def output_size(self):
        return len(self.y_columns)
    
    def _split_train_test(self, keys: list):
        for _ in range(int(self.train_test_ratio*len(keys))):
            self.train.append(keys.pop())
        self.test = keys

        if self.train_flag:
            self.keys = self.train
        else:
            self.keys = self.test


class S3WBDataset(WBDataset):

    def __init__(self, 
                 parquet_file, 
                 train_flag=True, 
                 train_test_ratio=.8, 
                 normalization="min_max"):
        super(S3WBDataset, self).__init__(
            train_flag=train_flag, 
            train_test_ratio=train_test_ratio,
            normalization=normalization
            )
    
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
        if self.norma is not None:
            self.norma.norm_x(X)
            self.norma.norm_y(y)
        return X, y
    

class FileWBDataset(WBDataset):

    def __init__(self, 
                parquet_file, 
                train_flag=True, 
                train_test_ratio=.8, 
                normalization="min_max"
                 ):
        super(FileWBDataset, self).__init__(
            train_flag=train_flag, 
            train_test_ratio=train_test_ratio,
            normalization=normalization
            )

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
        if self.norma is not None:
            self.norma.norm_x(X)
            self.norma.norm_y(y)
        return X, y
    

from azureml.fsspec import AzureMachineLearningFileSystem


class AzureDataset(WBDataset):
   
    def __init__(self, 
                 uri, 
                 train_flag=True, 
                 train_test_ratio=.8, 
                normalization="min_max"):
        super(AzureDataset, self).__init__(
            train_flag=train_flag, 
            train_test_ratio=train_test_ratio,
            normalization=normalization)

        self.fs = AzureMachineLearningFileSystem(uri)
     
        keys = self.fs.glob("**/*.parquet")
        
        self._split_train_test(keys)

   
    

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


from wb.utils.args import Args, FSType

def dataset(args: Args):
    fs_type, path = args.data_dir
    file_type, filename = args._filetype()
    if fs_type == FSType.FILE:
        train_dataset = FileWBDataset(
            os.path.join(path, filename), 
            train_flag=True, 
            train_test_ratio=args.ratio_train_test
            )
        test_dataset = FileWBDataset(
            os.path.join(path, filename), 
            train_flag=False, 
            train_test_ratio=args.ratio_train_test
            )

    return train_dataset, test_dataset