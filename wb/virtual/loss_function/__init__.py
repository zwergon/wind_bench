import torch
from torch import nn
from wb.virtual.loss_function.rmse import RMSELoss


def loss_function(config: dict, device):
     
    k = config['loss']
    if k == "MSE":
        criterion = nn.MSELoss()
    elif k == "RMSE":
        criterion = RMSELoss()
    elif k == "MAE":
        criterion = nn.SmoothL1Loss()
    return criterion