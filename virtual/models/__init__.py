

from virtual.models.LSTM_CNN import LSTMCNNModel
from virtual.models.LSTM import LSTMModel
from virtual.models.CNN import CNNModel
from virtual.models.MLP import MLPModel
from virtual.models.RNN  import RNNVanilla

def get_model(input_size, output_size, config: dict):
     
    kind = config['type']
   
    if kind == "LSTM":
        model = LSTMModel(
            input_size,
            config['hidden_size'],
            config['num_layers'], 
            output_size,
            config['dropout']
            )
    elif kind == "MLP":
        model = MLPModel(
            input_size, 
            output_size
        )
    elif kind == "CNN":
        model = CNNModel(
            input_size, 
            output_size,
            kernel_size=config['kernel_size']
            ) 
    elif kind == "RNN":
        model = RNNVanilla(
            input_size,
            output_size,
            config['hidden_size'],
            config['dropout']
        )
    elif kind == "LSTM_CNN":
        model = LSTMCNNModel(
            input_size,
            config['hidden_size'],
            config['num_layers'], 
            output_size,
            config['dropout']
            )
    
    else:
        raise Exception(f"model of type {kind} is not handled")

    return model

