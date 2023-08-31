import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, kernel_size):
        super(CNNModel, self).__init__()
        
        self.encodeur=nn.Sequential(
                      nn.Conv1d(input_size, 15, kernel_size=kernel_size),
                      nn.BatchNorm1d(15),
                      nn.ReLU(),
                      nn.Conv1d(15, 20, kernel_size=kernel_size),
                      nn.BatchNorm1d(20),
                      nn.ReLU(),
                      nn.Conv1d(20, 25, kernel_size=kernel_size),
                      nn.BatchNorm1d(25),
                      nn.ReLU()
                     )

        self.decodeur=nn.Sequential(
                      nn.ConvTranspose1d(25, 20, kernel_size=kernel_size),
                      nn.BatchNorm1d(20),
                      nn.ReLU(),
                      nn.ConvTranspose1d(20, 15, kernel_size=kernel_size),
                      nn.BatchNorm1d(15),
                      nn.ReLU(),
                      nn.ConvTranspose1d(15, output_size, kernel_size=kernel_size)
        )



    def forward(self, x):
        #x = x.permute(0, 2, 1)
        x = self.encodeur(x)
        x = self.decodeur(x)
        #x = x.permute(0, 2, 1)
        return x

    



