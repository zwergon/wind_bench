import os
import torch
from torch.utils.data import DataLoader

import ray
from ray import tune
from functools import partial

from wind_bench.dataset import NumpyWBDataset

from virtual.models import get_model
from virtual.Training import train_test

from virtual.logger import Logger
from virtual.checkpoint import CheckPoint

from args import Args

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #INPUT Parameters 
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))

    train_dataset = NumpyWBDataset(args.data_dir, indices=args.indices)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = NumpyWBDataset(args.data_dir, train_flag=False, indices=args.indices)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    #print(train_dataset.__getitem__(0, 7))
    train = train_dataset.__getitem__(0)[1].reshape((12000,6))
    #test =  test_dataset.__getitem__(0, 7)[1].reshape((1200000,6))
    #data = np.concatenate((train, test), axis=0)
    df = pd.DataFrame(train, 
                    columns=[
             "Mudline moment Mx[kNm]",
        "Mudline moment My[kNm]",
        "Mudline moment Mz[kNm]",
        "Waterline moment Mx[kNm]",
        "Waterline moment My[kNm]",
        "Waterline moment Mz[kNm]"
        ])
    print(df.describe()["Waterline moment Mz[kNm]"])

    #plt.plot(train_dataset.__getitem__(0, 7)[1][0,0,:])
    #plt.show()
    