import torch
import torch.nn as nn
import torch.optim as optim

class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout):
        super(CNNModel, self).__init__()
        
        self.encodeur=nn.Sequential(
                      nn.Conv1d(input_size, 7, kernel_size=kernel_size, padding="same"),
                      nn.BatchNorm1d(7),
                      nn.Dropout(dropout),
                      nn.ReLU(),
                      nn.Conv1d(7, 5, kernel_size=kernel_size, padding="same"),
                      nn.ReLU()
                     )

        padding = (kernel_size - 1) // 2
        self.decodeur=nn.Sequential(   
                      nn.ConvTranspose1d(5, 7, kernel_size=kernel_size, padding=padding),
                      nn.BatchNorm1d(7),
                      nn.ReLU(),
                      nn.ConvTranspose1d(7, output_size, kernel_size=kernel_size, padding=padding),
                     
        )



    def forward(self, x):
        x = self.encodeur(x)
        x = self.decodeur(x)
        return x

    



