


from virtual.models.LSTM import LSTMModel
from virtual.models.CNN import CNNModel
from virtual.models.MLP import MLPModel

def get_model(input_size, output_size, config: dict):
     
    kind = config['type']
   
    if kind == "LSTM":
        model = LSTMModel(
            input_size,
            config['hidden_size'],
            config['num_layers'], 
            output_size
            )
    elif type == "MLP":
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
    else:
        raise Exception(f"model of type {kind} is not handled")

    return model

