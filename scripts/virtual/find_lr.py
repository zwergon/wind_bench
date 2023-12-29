import os
import sys

from torch.utils.data import DataLoader

from wb.dataset import FileWBDataset
from wb.utils.config import Config
from wb.virtual.context import Context
from wb.virtual.training import find_lr
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
            train_dataset, batch_size=args.batch_size, num_workers=4
        )

        ctx.create_model(train_loader)

        find_lr(ctx, train_loader=train_loader)


if __name__ == "__main__":
    main()
