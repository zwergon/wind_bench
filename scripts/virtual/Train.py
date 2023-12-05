import os
import torch
from torch.utils.data import DataLoader
import sys

sys.path.append(os.getcwd())

from functools import partial

from wb.dataset import FileWBDataset
from wb.virtual.models import get_model
from wb.virtual.Training import train_test

from wb.virtual.context import Context
from wb.virtual.checkpoint import CheckPoint

from wb.utils.config import Config


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file", default="D:/lecomtje/Repositories/MPU/wb-virtual/tests/data/100_128\wind_bench.parquet")
    parser.add_argument("-c", "--config", help="training config file", 
                        type=str, 
                        default= os.path.join(os.path.dirname(__file__), "config.json")
                        )
    args = parser.parse_args()

    
    #INPUT Parameters
    config = Config(jsonname = args.config)

    with Context(config) as ctx:
     
        train_dataset = FileWBDataset(
            args.dataset, 
            train_flag=True, 
            train_test_ratio=config.ratio_train_test,
            indices=config.indices
            )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers
            )

        test_dataset = FileWBDataset(
            args.dataset, 
            train_flag=False, 
            train_test_ratio=config.ratio_train_test,
            indices=config.indices
            )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers
            )

        model = get_model(
            train_dataset.input_size,  
            train_dataset.output_size,
            config.__dict__)
        model = model.to(ctx.device)


        train_test(ctx, model, train_loader, test_loader)
