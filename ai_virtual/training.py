from torch import nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import time as timer


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

def optimiseur(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
 
def loss_function(device):
    return nn.MSELoss().to(device)

def train_test_model(model, train_loader, test_loader, config, device):
    model.to(device)
    model.train()
    loss_train = []
    loss_test = []
    criterion = loss_function(device)
    optimizer = optimiseur(model, config)

    for epoch in range(config['epoch']):

        start.record()
        start_cpu = timer.time()
        for (data_batch, y_batch) in train_loader:
            data_batch = data_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            prediction = model(data_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
        loss_train.append(loss.item())
        print(f'epoch= {epoch} loss = {loss: .3}')
        end.record()
        end_cpu = timer.time()
        # Waits for everything to finish running
        torch.cuda.synchronize()
        print(f"Time elapsed (GPU): {start.elapsed_time(end)}") 
        print(f"Time elapsed (Global): {end_cpu - start_cpu}") 

        model.eval()
        loss = 0.
        batch = 0
        for (data_batch, y_batch) in test_loader:
            batch+= 1
            data_batch = data_batch.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                prediction = model(data_batch)
                loss+= criterion(prediction, y_batch)
        loss/=batch
        loss_test.append(loss.item())
        
        print(f'valid loss = {loss: .3}')

    return y_batch, prediction, loss_train, loss_test

def train_test_plot(y_true, y_pred, loss_train, loss_test, config):

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(np.arange(config['epoch']), loss_train, label="loss_train")
    ax1.plot(np.arange(config['epoch']), loss_test, label="loss_test")
    ax1.legend()

    ax2.plot(y_true[0, :], label="y_true")
    ax2.plot(y_pred[0, :], label="y_pred")
    ax2.legend()
    plt.show()