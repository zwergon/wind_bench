import os
import torch
from torch.utils.data import DataLoader

from wb.dataset import AzureDataset

from wb.virtual.models import get_model
from wb.virtual.Training import train_test

from wb.virtual.logger import Logger
from wb.virtual.checkpoint import CheckPoint

from wb.utils.args import Args

if __name__ == "__main__":
    
    #INPUT Parameters
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))

    logger = Logger()
    logger.init(task_name=args.name, config=args.__dict__)

    checkpoint = CheckPoint(".", args.type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    train_dataset = AzureDataset(args.azure, train_flag=True, train_test_ratio=args.ratio_train_test)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = AzureDataset(args.azure, train_flag=False, train_test_ratio=args.ratio_train_test)
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
