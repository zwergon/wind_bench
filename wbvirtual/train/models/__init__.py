from wbvirtual.train.models.LSTM_CNN import LSTMCNNModel
from wbvirtual.train.models.LSTM import LSTMModel
from wbvirtual.train.models.CNN import CNNModel
from wbvirtual.train.models.MLP import MLPModel
from wbvirtual.train.models.RNN import RNNVanilla
from wbvirtual.train.models.UNet1D import UNet1D


models_dict = {
    "LSTM": LSTMModel,
    "MLP": MLPModel,
    "CNN": CNNModel,
    "RNN": RNNVanilla,
    "LSTM_CNN": LSTMCNNModel,
    "UNET1D": UNet1D,
}


class ModelError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def get_model(config: dict, input_size, output_size, device):
    kind = config["type"]
    try:
        model = models_dict[kind](input_size, output_size, config)
    except KeyError:
        raise ModelError(f"model {kind} is not handled")

    model = model.to(device=device)
    return model
