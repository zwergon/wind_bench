import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from wb.virtual.models import get_model
from wb.dataset import dataset

from wb.utils.config import Config
from wb.virtual.checkpoint import CheckPoint

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
        index = 0

        config = Config(jsonname = os.path.join(os.path.dirname(__file__), "config.json"))


        _, test_dataset = dataset(config)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        
        X_test, y_test = next(iter(test_loader))
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}")
    

        print(f"Type Network: {config.type}")
        model = get_model(test_dataset.input_size, test_dataset.output_size, config.__dict__)
        model.load_state_dict(get_state_dict(config))

        #real_outputs, predicted_outputs = get_all_predicted(model, test_loader, y_test)
        real_outputs, predicted_outputs = get_one_output(model, test_loader, index)

        # real_outputs = real_outputs - np.mean(real_outputs)
        # predicted_outputs = predicted_outputs - np.mean(predicted_outputs)

        # Plot actual vs. predicted values
        plt.figure(figsize=(9, 6))
        plt.plot(real_outputs[offset:], label='Actual')
        plt.plot(predicted_outputs[offset:], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Mudline Moment')
        plt.title('Actual vs. Predicted')
        plt.legend()
        plt.show()