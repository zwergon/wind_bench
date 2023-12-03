import torch
import numpy as np
from torch import nn

import torch.optim as optim
from wb.virtual.loss_function import get_loss
from wb.virtual.context import Context
from wb.utils.config import Config
from wb.virtual.prediction import Prediction
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler



try:
    from ray import tune
    from ray.air import  session
except ImportError:
    print("ray is not installed... set tune to False in config.json ")


from sklearn.metrics import r2_score

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.5
    return 0.995 ** epoch


def train_test(context: Context, model, train_loader, test_loader):

    device = context.device
    config : Config = context.config

    num_epochs = config["epoch"]

    context.summary(train_loader, test_loader)

    criterion = get_loss(device, config)

    optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config['weight_decay']
            )

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    for epoch in range(num_epochs):

        #Train
        model.train()
        train_loss = 0.0
        labels_list_train = []
        outputs_list_train = []

        labels_list_test = []
        outputs_list_test = []

        for X, Y in train_loader:
            
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)
            
            loss = criterion(Y_hat, Y)
            loss.backward()
         
            optimizer.step()
        
            train_loss += loss.item()
            labels_list_train.append(Y.cpu())
            outputs_list_train.append(Y_hat.detach().cpu())
        
        scheduler.step()
        
        train_loss /= len(train_loader)

        all_labels = torch.cat(labels_list_train, dim=0)
        all_outputs= torch.cat(outputs_list_train, dim=0)
            
        mae_train =  nn.L1Loss()(all_labels, all_outputs)

        all_labels_flat = all_labels.numpy().flatten()
        all_outputs_flat = all_outputs.numpy().flatten()

        r2_train = r2_score(all_labels_flat, all_outputs_flat)
        # Test
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for X, Y in test_loader:
                X = X.to(device)
                Y = Y.to(device)
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
                
                test_loss += loss.item()
                
                labels_list_test.append(Y.cpu())
                outputs_list_test.append(Y_hat.detach().cpu())
        
        test_loss /= len(test_loader)
   
        all_labels_test = torch.cat(labels_list_test, dim=0)
        all_outputs_test= torch.cat(outputs_list_test, dim=0)
        
        all_labels_test_flat = all_labels_test.numpy().flatten()
        all_outputs_test_flat = all_outputs_test.numpy().flatten()

        mae_test =  nn.L1Loss()(all_labels_test, all_outputs_test)
        
        r2_test = r2_score(all_labels_test_flat, all_outputs_test_flat)
        
        if config["tune"] == True:
           session.report({"r2_score": r2_test})

        # Reporting
        losses = {
            "Loss": (train_loss, test_loss), 
            "MAE": (mae_train.item(), mae_test.item()),
            "R2": (r2_train.item(), r2_test.item())
        }
        
        context.report_loss(epoch, losses)

        if epoch % 10 == 0:
            predictions = [
                Prediction(f"results_{epoch}.png", labels_list_test[0], outputs_list_test[0])
            ]
            context.report_prediction(predictions)

            context.save_checkpoint(epoch=epoch, 
                                    model=model, 
                                    optimizer=optimizer, 
                                    loss=test_loss
                                    )

       
