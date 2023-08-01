import torch
import pytorch_lightning as pl
from ai.dataset import FSBDataset
from torch.utils.data import random_split, DataLoader

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
  
    def setup(self, stage):
        #multi gpu
        all_dataset = FSBDataset("D:/work/ai.virtual/data/five_story_building_ts_with_us_1000.npy")
        dataset = torch.utils.data.Subset(all_dataset, range(48000))
        
        train_val_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_val_size
        train_val_dataset, self.test_dataset = random_split(dataset, [train_val_size, test_size])

        train_size= int(0.8*len(train_val_dataset))
        val_size=len(train_val_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(train_val_dataset, [train_size, val_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['batch_size'], num_workers=self.config['num_workers'], shuffle=False)

    
        
