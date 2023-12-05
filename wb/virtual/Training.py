import torch
import numpy as np
from torch import nn

import torch.optim as optim

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

from torchmetrics import R2Score

from wb.dataset import WBDataset
from wb.virtual.loss_function import get_loss
from wb.virtual.context import Context
from wb.utils.config import Config
from wb.virtual.predictions import Predictions
from wb.virtual.metrics_collection import MetricsCollection

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.5
    return 0.995 ** epoch


def train_test(context: Context, model, train_loader, test_loader):

    device = context.device
    config : Config = context.config
    train_dataset : WBDataset = train_loader.dataset

    num_epochs = config["epoch"]

    context.summary(train_loader, test_loader)

    criterion = get_loss(device, config)

    optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config['weight_decay']
            )


    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    predictions = Predictions(train_loader, device )

    for epoch in range(num_epochs):


        #Train
        train_loss = 0.0
        train_metrics = MetricsCollection(train_dataset.output_size, device)

        model.train()
        for X, Y in train_loader:
            
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)

            loss = criterion(Y_hat, Y)
            loss.backward()
         
            optimizer.step()

            train_metrics.update_from_batch(Y_hat, Y)

            train_loss += loss.item()
        
        scheduler.step()
        
        train_loss /= len(train_loader)
        train_metrics.compute()

        # Test
        test_loss = 0.0
        test_metrics = MetricsCollection(train_dataset.output_size, device)
        model.eval()
        
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                
                test_loss += loss.item()
                test_metrics.update_from_batch(Y_hat, Y)
                
        test_loss /= len(test_loader)
        test_metrics.compute()
        
        # Reporting
        context.report_loss(epoch, train_loss, test_loss)


        context.report_metrics(epoch, train_metrics)
        context.report_metrics(epoch, test_metrics)
        
    
        if epoch % 10 == 0:
            
            predictions.compute(epoch, model)
            
            context.report_prediction(predictions)

            context.save_checkpoint(epoch=epoch, 
                                    model=model, 
                                    optimizer=optimizer, 
                                    loss=test_loss
                                    )

       
