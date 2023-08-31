import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from wind_bench.dataset import NumpyWBDataset
from virtual.models import get_model
from virtual.Training import train_test

from args import Args


if __name__ == "__main__":
    
    #INPUT Parameters
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_dataset = NumpyWBDataset(args.data_dir, indices=[0])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    test_dataset = NumpyWBDataset(args.data_dir, train_flag=False, indices=[0])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    X_train, y_train = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))
    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")
    

    print(f"Type Network: {args.type}")
    model = get_model(train_dataset.input_size, train_dataset.output_size, args.__dict__)
    model = model.to(device)
        
    train_test(model, train_loader, test_loader, args.__dict__, device)

    torch.save(model.state_dict(), os.path.join(args.data_dir, f"model_{args.type.lower()}.pth"))
