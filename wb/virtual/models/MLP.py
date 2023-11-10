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
        x=x.view(x.size(0)*x.size(2),x.size(1))
        x = self.encodeur(x)
        x=x.view(-1,x.size(1),2000)
        return x

    



