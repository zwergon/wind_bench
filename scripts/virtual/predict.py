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

def get_all_predicted(model, test_loader, y_test):
        all_outputs = []
        # Step 9: Perform prediction on the test set
        model.eval()
        with torch.no_grad():
                for inputs, _ in test_loader:
                        outputs = model(inputs)
                        all_outputs.append(outputs)

        # Combine all the outputs into a single array
        predicted_outputs = torch.cat(all_outputs, dim=0)

        # Convert predicted_outputs to a numpy array
        predicted_outputs = predicted_outputs.numpy()

        # Reshape predicted_outputs and y_test for plotting
        predicted_outputs = predicted_outputs.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)


        return y_test, predicted_outputs


def get_one_output(model, test_loader, index):
        model.eval()
        inputs, actual = next(iter(test_loader))
        with torch.no_grad():
                predicted = model(inputs)

        return actual[index, 0, :].numpy(), predicted[index, 0, :].numpy()


def get_state_dict(args):

        checkpoint = CheckPoint.load(args.checkpoint)
       
        return checkpoint.state_dict

if __name__ == "__main__":

        offset = 10
      
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", help="path to parquet file")
        parser.add_argument("checkpoint", help="path to checkpoint")
        parser.add_argument("index", help="which element in the dataset", type=int)
        parser.add_argument("-s", "--span", help="range in dataset to keep [1000:1500]", type=int, nargs='+', default=[1000, 1500])
        parser.add_argument("-i", "--indices", help="output indices [0..6] (Default:0)", nargs='+', type=int, default=[0])
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
        
        X_test, y_test = next(iter(test_loader))
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}")
    

        print(f"Type Network: {config.type}")
        model = get_model(ctx, dataset.input_size, dataset.output_size)
        
        model.load_state_dict(checkpoint.state_dict)

        real_outputs, predicted_outputs = get_one_output(model, test_loader, args.index)

        # Plot actual vs. predicted values
        plt.figure(figsize=(9, 6))
        plt.plot(real_outputs[args.span[0]:args.span[1]], label='Actual')
        plt.plot(predicted_outputs[args.span[0]:args.span[1]], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel(dataset.output_name(args.index))
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()