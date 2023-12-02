import torch
import numpy as np
from torch import nn

import torch.optim as optim
from wb.virtual.checkpoint import CheckPoint
from wb.virtual.loss_function import get_loss

import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

try:
    from ray import tune
    from ray.air import  session
except ImportError:
    print("ray is not installed... set tune to False in args.json ")


from sklearn.metrics import r2_score

def lr_lambda(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.5
    return 0.995 ** epoch

def train_test(config, model, train_loader, test_loader, context: dict):

    logger = context['logger']
    cf = context['config']
    device = context['device']
    checkpoint: CheckPoint = context['checkpoint']


    num_epochs = cf["epoch"]

    logger.summary(device, train_loader, test_loader)

    
    optimizer = optim.Adam(
            model.parameters(),
            lr=cf["learning_rate"],
            weight_decay=cf['weight_decay']
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

        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = get_loss(device, cf)(outputs, labels)
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
        
            train_loss += loss.item()
            labels_list_train.append(labels.cpu())
            outputs_list_train.append(outputs.detach().cpu())
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        train_loss /= len(train_loader)

        all_labels = torch.cat(labels_list_train, dim=0)
        all_outputs= torch.cat(outputs_list_train, dim=0)
            
        mae_train =  nn.L1Loss()(all_labels, all_outputs)

        all_labels_flat = all_labels.numpy().reshape(-1, cf["sequence_length"])
        all_outputs_flat = all_outputs.numpy().reshape(-1, cf["sequence_length"])

        r2_train = r2_score(all_labels_flat, all_outputs_flat)
        # Test
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = get_loss(device, cf)(outputs, labels)
                
                test_loss += loss.item()
                
                labels_list_test.append(labels.cpu())
                outputs_list_test.append(outputs.detach().cpu())
        
        test_loss /= len(test_loader)
   
        all_labels_test = torch.cat(labels_list_test, dim=0)
        all_outputs_test= torch.cat(outputs_list_test, dim=0)
        
        all_labels_test_flat = all_labels_test.numpy().reshape(-1, cf["sequence_length"])
        all_outputs_test_flat = all_outputs_test.numpy().reshape(-1, cf["sequence_length"])

        mae_test =  nn.L1Loss()(all_labels_test, all_outputs_test)
        
        r2_test = r2_score(all_labels_test_flat, all_outputs_test_flat)
        
        if cf["tune"] == True:
           session.report({"r2_score": r2_test})

        # Reporting
        if epoch % 100 == 0:
            index = 0
            offset = 30
            inputs, actual = next(iter(test_loader))
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predicted = model(inputs)

            
            checkpoint.save(epoch=epoch, model=model, optimizer=optimizer, loss=test_loss)

        losses = {
            "Loss": (train_loss, test_loss), 
            "MAE": (mae_train.item(), mae_test.item()),
            "F2": (r2_train.item(), r2_test.item())
        }
        logger.report_loss(epoch, losses)


"""    
    plt.figure()
    plt.plot(all_labels[0,0,:], label="y_true")
    plt.plot(all_outputs[0,0,:], label="y_pred")
    plt.legend()
    plt.show()
"""   
