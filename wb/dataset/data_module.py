import os
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from wb.dataset import S3WBDataset, FileWBDataset, NumpyWBDataset


class WBDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
  
    def setup(self, stage):
        
        config = self.config

        if stage == "fit":
            if config['dataset'] == "S3":
                self.train_dataset = S3WBDataset(config['filename'], train_flag=True)
            elif config['dataset'] == "Parquet":
                self.train_dataset = FileWBDataset(config['filename'], train_flag=True)
            elif config['dataset'] == "numpy":
                data_dir = os.path.join(config['root_path'], f"{config['n_samples']}_{config['sequence_length']}")
                self.train_dataset = NumpyWBDataset(data_dir, train_flag=True, indices=config['indices'])
            else:
                raise Exception(f"dataset {stage} : {config['dataset']} not handled")


        if stage == "test":
            if config['dataset'] == "S3":
                self.test_dataset = S3WBDataset(config['filename'], train_flag=False)
            elif config['dataset'] == "Parquet":
                self.test_dataset = FileWBDataset(config['filename'], train_flag=False)
            elif config['dataset'] == "numpy":
                data_dir = os.path.join(config['root_path'], f"{config['n_samples']}_{config['sequence_length']}")
                self.test_dataset = NumpyWBDataset(data_dir, train_flag=False, indices=config['indices'])
            else:
                raise Exception(f"dataset {stage} : {config['dataset']} not handled")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.config['batch_size'], 
                          num_workers=self.config['num_workers'], 
                          shuffle=True
                          )
    
    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                          batch_size=self.config['batch_size'], 
                          num_workers=self.config['num_workers'], 
                          shuffle=False
                          )
    
   
        
