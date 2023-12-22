import os
import argparse
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from wb.utils.config import Config
from wb.utils.display import predictions_plot
from wb.dataset import FileWBDataset
from wb.virtual.context import Context
from wb.virtual.checkpoint import CheckPoint
from wb.virtual.models import get_model
from wb.virtual.predictions import Predictions
from wb.virtual.feature import Feature
from wb.post.eq_load import eq_load

from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file")
    parser.add_argument("checkpoint", help="path to checkpoint")
    parser.add_argument("index", help="which element in the dataset", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-s", "--span", help="range in dataset to keep [1000:1500]", type=int, nargs="+"
    )
    group.add_argument(
        "-a", "--all", help="display all range of signal", action="store_true"
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
        default=os.path.join(os.path.dirname(__file__), "config.json"),
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

    model = get_model(ctx, test_dataset.input_size, test_dataset.output_size)
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


if __name__ == "__main__":
    main()
