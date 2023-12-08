
import os
import torch
import sys

sys.path.append(os.getcwd())

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from wb.dataset import FileWBDataset
from wb.utils.config import Config
from wb.virtual.context import Context
from wb.virtual.training import find_lr
from wb.virtual.models import get_model
from arguments import parse_args
 



import numpy as np

if __name__ == "__main__":

    args = parse_args()
    
    #INPUT Parameters
    config = Config.create_from_args(args)

    with Context(config) as ctx:
     
        train_dataset = FileWBDataset(
            args.dataset, 
            train_flag=True, 
            train_test_ratio=config.ratio_train_test,
            indices=args.indices
            )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=
            args.batch_size, 
            num_workers=4
            )
        
        model = get_model(
            ctx,
            train_dataset.input_size,  
            train_dataset.output_size
            )

    
        find_lr(ctx, model, train_loader=train_loader)

       
