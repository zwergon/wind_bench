import os
import mlflow
from mlflow.models import infer_signature
import torch
from torch import nn
import argparse

from wb.utils.config import Config
from wb.dataset import FileWBDataset
from wb.virtual.models import get_model
from wb.virtual.context import Context

from torch.utils.data import DataLoader


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file")
    parser.add_argument("index", help="which element in the dataset", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s", "--span", help="range in timeserie to keep [S1, S2]", type=int, nargs=2
    )
    group.add_argument(
        "-a", "--all", help="display all timeserie (fullrange)", action="store_true"
    )
    parser.add_argument(
        "-i",
        "--indices",
        help="output indices [0..6] (Default:0)",
        nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "-c",
        "--config",
        help="training config file",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "config.json"),
    )
    parser.add_argument(
        "-n",
        "--norm",
        help="display normalized",
        choices=["none", "min_max"],
        default="min_max",
    )
    args = parser.parse_args()

    # INPUT Parameters
    config = Config.create_from_args(args)
    ctx = Context(config)

    dataset = FileWBDataset(
        args.dataset,
        train_flag=True,
        train_test_ratio=config.ratio_train_test,
        indices=args.indices,
        sensor_desc=config.sensors,
        normalization=args.norm,
    )

    loader = DataLoader(dataset, batch_size=1)

    print(f"train dataset (size) {len(dataset)}")

    X_idx, y_idx = next(iter(loader))
    print(X_idx.shape, y_idx.shape)

    model = get_model(ctx, dataset.input_size, dataset.output_size)
    
    
    with mlflow.start_run() as run:
        signature = infer_signature(X_idx.numpy(), y_idx.numpy())
        print(signature)
        model_info = mlflow.pytorch.log_model(model, "model", signature=signature)

    print(model_info.model_uri)

        
    pytorch_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    predictions = pytorch_pyfunc.predict(X_idx.numpy())
    print(predictions)

    model.eval()
    predict2 = model(X_idx)
    print(predict2.detach().numpy())
    print(predict2.detach().numpy() -predictions)