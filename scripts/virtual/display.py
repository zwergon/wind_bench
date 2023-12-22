import os
import sys
import argparse

import matplotlib.pyplot as plt

from wb.utils.config import Config
from wb.dataset import FileWBDataset

sys.path.append(os.getcwd())


def main():
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

    dataset = FileWBDataset(
        args.dataset,
        train_flag=True,
        train_test_ratio=config.ratio_train_test,
        indices=args.indices,
        sensor_desc=config.sensors,
        normalization=args.norm,
    )

    print(f"train dataset (size) {len(dataset)}")

    X_idx, y_idx = dataset[args.index]

    print(f"X_test : {X_idx.shape}")
    print(f"y_test : {y_idx.shape}")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    fig.suptitle(f"X (solid) versus y (dash) for idx:{args.index}")

    if args.all:
        deb = 0
        end = X_idx.shape[1] - 1
    else:
        deb = args.span[0]
        end = args.span[1]

    for i in range(X_idx.shape[0]):
        ax1.plot(X_idx[i, deb:end], label=f"{dataset.x_columns[i]}")
    ax1.legend()

    for i in range(y_idx.shape[0]):
        ax2.plot(y_idx[i, deb:end], "--", label=f"{dataset.y_selected[i]}")
    ax2.legend()

    plt.show()


if __name__ == "__main__":
    main()
