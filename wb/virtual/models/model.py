import torch
from torch import nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from wb.virtual.models import get_model

class LitModel(pl.LightningModule):
    def __init__(self, input_size=8, output_size=6, config={}):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.model = get_model(input_size, output_size, config)
       
    def forward(self, x): 
        return self.model(x)
    
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
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'], weight_decay=self.config['weight_decay'])
    
