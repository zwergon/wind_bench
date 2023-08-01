import torch
from torch import nn as nn
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encodeur=nn.Sequential(
                      nn.Conv1d(config['input_channels'], 32, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.Conv1d(32, 16, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.Conv1d(16, 8, kernel_size=config['kernel_size']) 
                            )
        self.decodeur=nn.Sequential(
                      nn.ConvTranspose1d(8, 16, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.ConvTranspose1d(16, 32, kernel_size=config['kernel_size']),
                      nn.ReLU(),
                      nn.ConvTranspose1d(32, config['output_channels'], kernel_size=config['kernel_size'])        
        )
        self.loss_fn = nn.MSELoss()
        self.config = config
    
    def forward(self, x): 
        x = self.encodeur(x)
        x = self.decodeur(x)
        x = x.squeeze(dim=1)
        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
    

    


