from wb.virtual.models.LSTM_CNN import LSTMCNNModel
from wb.virtual.models.LSTM import LSTMModel
from wb.virtual.models.CNN import CNNModel
from wb.virtual.models.MLP import MLPModel
from wb.virtual.models.RNN import RNNVanilla
from wb.virtual.models.UNet1D import UNet1D


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
