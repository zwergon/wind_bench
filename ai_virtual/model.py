from torch import nn as nn

class Conv1dAutoEncodeur(nn.Module):
    def __init__(self, config):
        super(Conv1dAutoEncodeur, self).__init__()
        self.encodeur=nn.Sequential(
                      nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.Conv1d(32, 16,kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.Conv1d(16, 8,kernel_size=config['kernel_size']) 
                            )
        self.decodeur=nn.Sequential(
                      nn.ConvTranspose1d(8, 16, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.ConvTranspose1d(16, 32, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.ConvTranspose1d(32, config['output_channels'], kernel_size=config['kernel_size'])        
        )

    def forward(self, x): 
        x = self.encodeur(x)
        x = self.decodeur(x)
        x = x.squeeze(dim=1)
        return x

class MultiChannelModel(nn.Module):
    def __init__(self, config ):
        super(MultiChannelModel, self).__init__()
        self.encodeur = nn.Sequential(
                      nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size']),
                      nn.LeakyReLU(0.01),
                      nn.Conv1d(32, 16,kernel_size=config['kernel_size']),
                      nn.LeakyReLU(0.01),
                      nn.Conv1d(16, 8,kernel_size=config['kernel_size']),
                      nn.LeakyReLU(0.01),
                      nn.MaxPool1d(2, return_indices=True),
                            )

        self.decodeur = nn.Sequential(
                      nn.ConvTranspose1d(8, 16, kernel_size=config['kernel_size']),
                      nn.LeakyReLU(0.01),
                      nn.ConvTranspose1d(16, 32, kernel_size=config['kernel_size']),
                      nn.LeakyReLU(0.01),
                      nn.ConvTranspose1d(32, config['output_channels'], kernel_size=config['kernel_size'])     
        )
        self.poolinverse = nn.MaxUnpool1d(2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x, indices = self.encodeur(x)
        x = self.poolinverse(x, indices)
        x = self.decodeur(x)
        x = self.flatten(x)
        return x

def get_model(config):
    if config['model_name']=="ConvAutoEnc":
        return Conv1dAutoEncodeur(config)
    else:
        return MultiChannelModel(config)