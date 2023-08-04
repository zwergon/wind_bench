import torch
from torch import nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

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
        self.y_true = None
        self.y_pred =  None

    
    def forward(self, x): 
        x = self.encodeur(x)
        x = self.decodeur(x)
        return x
    
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x=x.float()
        y=y.float()
        y_pred = self.forward(x)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred, y
    
    def training_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        mae = nn.L1Loss()(y.cpu(), y_pred.detach().cpu())
        self.log('train_loss', loss, prog_bar=True)
        self.log('MAE_train', mae, prog_bar=False)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        mae = nn.L1Loss()(y.cpu(), y_pred.cpu())
        self.log('val_loss', loss, prog_bar=True)
        self.log('MAE_test', mae, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, y_pred, y = self._common_step(batch, batch_idx)
        self.y_pred = y_pred.cpu().numpy()
        self.y_true =  y.cpu().numpy()
        self.log('test_loss', loss, prog_bar=True)
        return loss

    
    def on_test_end(self):
        
        plt.figure()
        plt.plot(self.y_true[0,0,:], label="y_true")
        plt.plot(self.y_pred[0,0,:], label="y_pred")
        plt.set_title("variable_1_ batch_1")
        plt.show()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
    
