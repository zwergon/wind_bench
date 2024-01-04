import argparse
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from wbvirtual.utils.config import Config
from wbvirtual.utils.arguments import parse_args
from wbvirtual.utils.display import predictions_plot
from wbvirtual.dataset import FileWBDataset
from wbvirtual.train.context import Context
from wbvirtual.train.checkpoint import CheckPoint
from wbvirtual.train.training import train_test, find_lr
from wbvirtual.train.predictions import Predictions
from wbvirtual.train.feature import Feature
from wbvirtual.post.eq_load import eq_load
from wbvirtual.train.metric_plot import MetricsBoxPlot

from sklearn.metrics import r2_score


def main_display():
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
        default=Config.default_path(),
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


def main_findlr():
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


def main_predict():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file")
    parser.add_argument("checkpoint", help="path to checkpoint")
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
        "-n",
        "--norm",
        help="display prediction normalized or not",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--config",
        help="training config file",
        type=str,
        default=Config.default_path(),
    )
    args = parser.parse_args()

    # INPUT Parameters
    config = Config.create_from_args(args)
    config.cuda = False

    checkpoint = CheckPoint.load(args.checkpoint)
    ctx = Context(config, checkpoint=checkpoint)

    test_dataset = FileWBDataset(
        args.dataset,
        train_flag=False,
        train_test_ratio=config.ratio_train_test,
        indices=args.indices,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    predictions = Predictions(test_loader, norm=args.norm)

    _, y_test = next(iter(test_loader))
    print(f"Number of predictions : {len(test_dataset)}")
    print(f"Shape of predictions : {y_test.shape}")
    print(f"Type Network: {config.type}")

    model = ctx.create_model(test_loader)
    model.load_state_dict(checkpoint.state_dict)

    predictions.compute(model, ctx.device)

    if args.all:
        deb = 0
        end = predictions.actual.shape[2]
    else:
        deb = args.span[0]
        end = args.span[1]

    predictions_plot(predictions, index=args.index, span=[deb, end])
    plt.show()

    feature = Feature(eq_load)
    feature.compute(predictions, m=3)

    print(f"r2score for DEL: {r2_score(feature.actual, feature.predicted)}")
    plt.scatter(feature.actual, feature.predicted)
    plt.show()

    r2 = MetricsBoxPlot(r2_score)
    r2.compute(predictions)
    plt.show()


def main_train():
    args = parse_args()

    # INPUT Parameters
    config = Config.create_from_args(args)

    with Context(config) as ctx:
        train_dataset = FileWBDataset(
            args.dataset,
            train_flag=True,
            train_test_ratio=config.ratio_train_test,
            indices=args.indices,
            sensor_desc=config.sensors,
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
            sensor_desc=config.sensors,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        ctx.create_model(train_loader)

        train_test(ctx, train_loader, test_loader)
