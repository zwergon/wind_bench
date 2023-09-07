import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from wind_bench.dataset import NumpyWBDataset
from virtual.models import get_model
from virtual.Training import train_test

from virtual.logger import Logger
from virtual.checkpoint import CheckPoint

from args import Args


if __name__ == "__main__":
    
    #INPUT Parameters
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))

    logger = Logger()
    logger.init(task_name=args.name, config=args.__dict__)

    checkpoint = CheckPoint(args.data_dir, args.type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    train_dataset = NumpyWBDataset(args.data_dir, indices=[0])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = NumpyWBDataset(args.data_dir, train_flag=False, indices=[0])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(
        train_dataset.input_size, 
        train_dataset.output_size, 
        args.__dict__)
    model = model.to(device)

    context = {
        "config": args.__dict__,
        "device": device,
        "logger": logger,
        "checkpoint": checkpoint
    }

    train_test(model, train_loader, test_loader, context)

    