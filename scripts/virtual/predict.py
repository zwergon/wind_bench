import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from wb.virtual.models import get_model
from wb.dataset import dataset

from wb.utils.config import Config
from wb.virtual.context import Context
from wb.virtual.checkpoint import CheckPoint
from wb.dataset import FileWBDataset

from sklearn.metrics import r2_score

def get_all_predicted(model, test_loader, norm):

        dataset : FileWBDataset = test_loader.dataset
       
        _, Y = dataset[0]
        predicted = np.zeros(shape=(len(dataset), Y.shape[0], Y.shape[1]))
        actual  = np.zeros(shape=(len(dataset), Y.shape[0], Y.shape[1]))
        idx = 0
        model.eval()
        with torch.no_grad():
                for X, Y in test_loader:
                        Y_hat = model(X)
                        if not norm and dataset.norma:
                                dataset.norma.unnorm_y(Y_hat)
                                dataset.norma.unnorm_y(Y)

                        for i in range(Y.shape[0]):
                                predicted[idx, :, :] = Y_hat[i, :, :]
                                actual[idx, :, :] = Y[i, :, :]
                                idx = idx + 1

       
        return predicted, actual


if __name__ == "__main__":

      
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", help="path to parquet file")
        parser.add_argument("checkpoint", help="path to checkpoint")
        parser.add_argument("index", help="which element in the dataset", type=int)
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument("-s", "--span", help="range in dataset to keep [1000:1500]", type=int, nargs='+')
        group.add_argument("-a", "--all", help="display all range of signal", action='store_true')
        parser.add_argument("-i", "--indices", help="output indices [0..6] (Default:0)", nargs='+', type=int, default=[0])
        parser.add_argument("-n", "--norm", help="get predicted normalized", action='store_true', default=False)
        parser.add_argument("-c", "--config", help="training config file", 
                                type=str, 
                                default= os.path.join(os.path.dirname(__file__), "config.json")
                                )
        args = parser.parse_args()

        
        #INPUT Parameters
        config = Config.create_from_args(args)
        config.cuda = False

        checkpoint = CheckPoint.load(args.checkpoint)
        ctx = Context(config, checkpoint=checkpoint)


        dataset = FileWBDataset(
            args.dataset, 
            train_flag=False, 
            train_test_ratio=config.ratio_train_test,
            indices=args.indices
            )
        test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        _, y_test = next(iter(test_loader))
        print(f"Number of predictions : {len(dataset)}")
        print(f"Shape of predictions : {y_test.shape}")
        print(f"Type Network: {config.type}")


        model = get_model(ctx, dataset.input_size, dataset.output_size)  
        model.load_state_dict(checkpoint.state_dict)

        predicted, actual = get_all_predicted(model, test_loader=test_loader, norm=args.norm)

        if args.all:
            deb = 0
            end = actual.shape[2]
        else:
            deb = args.span[0]
            end = args.span[1]

      
        
        fig, axs =  plt.subplots(len(args.indices), sharex=True, squeeze=False)
        print(axs)
        fig.suptitle('Actual vs. Predicted')
        for i, idx in enumerate(args.indices):
                y = actual[args.index, i, deb:end]
                y_hat = predicted[args.index, i, deb:end]

                print(f"r2_score for {dataset.output_name(idx)}: {r2_score(y, y_hat)}")
                axs[i][0].plot(y, label='Actual')
                axs[i][0].plot(y_hat, label='Predicted')
                axs[i][0].set_ylabel(dataset.output_name(idx))

        plt.show()