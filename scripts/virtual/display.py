
import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt

from wb.utils.config import Config
from wb.dataset import FileWBDataset
from wb.virtual.context import Context

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to parquet file")
    parser.add_argument("index", help="which element in the dataset", type=int)
    parser.add_argument("-s", "--span", help="range in dataset to keep [1000:1500]", type=int, nargs='+', default=[1000, 1500])
    parser.add_argument("-i", "--indices", help="output indices [0..6] (Default:0)", nargs='+', type=int, default=[0])
    parser.add_argument("-c", "--config", help="training config file", 
                        type=str, 
                        default= os.path.join(os.path.dirname(__file__), "config.json")
                        )
    args = parser.parse_args()

    #INPUT Parameters
    config = Config(args)

    with Context(config) as ctx:
     
        dataset = FileWBDataset(
            args.dataset, 
            train_flag=True, 
            train_test_ratio=config.ratio_train_test,
            indices=args.indices
            )
        

        print(f"train dataset (size) {len(dataset)}")

        X_idx, y_idx = dataset[args.index]

        print(f"X_test : {X_idx.shape}")
        print(f"y_test : {y_idx.shape}")

        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle( f"X (solid) versus y (dash) for idx:{args.index}")

        for i in range(X_idx.shape[0]):
            ax1.plot(X_idx[i, args.span[0]:args.span[1]])

        for i in range(y_idx.shape[0]):
            ax2.plot(y_idx[i, args.span[0]:args.span[1]], "--")

    
        plt.show()

    