from torch import nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt


def optimiseur(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
 
def loss_function(device):
    return nn.MSELoss().to(device)

def train_test_model(model, train_loader, test_loader, config, device):
    model=model.to(device)
    model.train()
    losses_train = []
    losses_test = []
    criterion = loss_function(device)
    optimizer = optimiseur(model, config)

    for epoch in range(config['epoch']):
        for (data_batch, y_batch) in train_loader:
            data_batch = data_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            optimizer.zero_grad()
            prediction = model(data_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
        losses_train.append(loss.item())
        print(f'epoch= {epoch} loss = {loss: .3}')

        model.eval()
        loss = 0.
        for (data_batch, y_batch) in test_loader:
            data_batch = data_batch.float().to(device)
            y_batch = y_batch.float().to(device)
            with torch.no_grad():
                prediction = model(data_batch)
                loss+= criterion(prediction, y_batch)
        losses_test.append(loss.item())
        print(f'valid loss = {loss: .3}')
    
    plt.figure()
    plt.plot(np.arange(config['epoch']), losses_train, label="loss_train")
    plt.plot(np.arange(config['epoch']), losses_test, label="loss_test")
    plt.legend()
    plt.show()

