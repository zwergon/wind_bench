import torch
from torch import nn
from wb.virtual.loss_function.dilate import Dilate
from wb.virtual.loss_function.rmse import RMSELoss

def loss_function(config: dict, device):
     
    k = config['loss']
    if k == "MSE":
        criterion = nn.MSELoss()

    elif k == "dilate":
        criterion = Dilate(device, config).forward
    
    elif k == "RMSE":
        criterion = RMSELoss()
    return criterion