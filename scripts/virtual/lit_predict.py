import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from wb.virtual.models import get_model
from wb.virtual.models.model import LitModel
from wb.dataset import NumpyWBDataset

from wb.utils.config import Config


def get_one_output(model, test_loader, index):
        model.eval()
        inputs, actual = next(iter(test_loader))
        with torch.no_grad():
                predicted = model(inputs)

        return actual[index, 0, :].numpy(), predicted[index, 0, :].numpy()


if __name__ == "__main__":

        config = Config(jsonname = os.path.join(os.path.dirname(__file__), "config.json"))


        test_dataset = NumpyWBDataset(config.data_dir, train_flag=False)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        X_test, y_test = next(iter(test_loader))
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}")
    

        print(f"Type Network: {config.type}")
        print("checkpoint: ", os.path.join(config.data_dir, config.state_dict))
       
        model = LitModel.load_from_checkpoint(os.path.join(config.data_dir, config.state_dict))
        model.to(X_test.device)
        real_outputs, predicted_outputs = get_one_output(model, test_loader, config.index)

        # real_outputs = real_outputs - np.mean(real_outputs)
        # predicted_outputs = predicted_outputs - np.mean(predicted_outputs)

        # Plot actual vs. predicted values
        plt.figure(figsize=(9, 6))
        plt.plot(real_outputs, label='Actual')
        plt.plot(predicted_outputs, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Mudline Moment')
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()