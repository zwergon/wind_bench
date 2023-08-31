import torch
import torch.nn as nn
import torch.optim as optim

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLPModel, self).__init__()
        
        self.encodeur=nn.Sequential(
                      nn.Linear(input_size, 512),
                      nn.ReLU(),
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Linear(256, 128),
                      nn.ReLU(), 
                      nn.Linear(128, 256),
                      nn.ReLU(),
                      nn.Linear(256, 512),
                      nn.ReLU(),
                      nn.Linear(512, output_size)
                    )
        
    def forward(self, x):
        x = self.encodeur(x)
        return x

    



