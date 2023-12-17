import os
import sys

from torch.utils.data import DataLoader

from wb.dataset import FileWBDataset
from wb.virtual.models import get_model
from wb.virtual.training import train_test
from wb.virtual.context import Context
from wb.utils.config import Config

from scripts.virtual.arguments import parse_args

sys.path.append(os.getcwd())


def main():
    args = parse_args()

    # INPUT Parameters
    config = Config.create_from_args(args)

    with Context(config) as ctx:
        train_dataset = FileWBDataset(
            args.dataset,
            train_flag=True,
            train_test_ratio=config.ratio_train_test,
            indices=args.indices,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

        test_dataset = FileWBDataset(
            args.dataset,
            train_flag=False,
            train_test_ratio=config.ratio_train_test,
            indices=args.indices,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        model = get_model(ctx, train_dataset.input_size, train_dataset.output_size)

        train_test(ctx, model, train_loader, test_loader)


if __name__ == "__main__":
    main()
