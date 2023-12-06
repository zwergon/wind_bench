


from wb.virtual.context import Context
from wb.virtual.models.LSTM_CNN import LSTMCNNModel
from wb.virtual.models.LSTM import LSTMModel
from wb.virtual.models.CNN import CNNModel
from wb.virtual.models.MLP import MLPModel
from wb.virtual.models.RNN  import RNNVanilla

def get_model(context: Context, input_size, output_size):
     
    config = context.config

    kind = config['type']
   
    if kind == "LSTM":
        model = LSTMModel(
            input_size,
            config['hidden_size'],
            config['num_layers'], 
            output_size
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
            kernel_size=config['kernel_size'],
            dropout=config['dropout']
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

    model = model.to(device=context.device)
    return model

