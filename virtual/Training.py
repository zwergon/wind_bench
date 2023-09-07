import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from virtual.checkpoint import CheckPoint

def train_test(model, train_loader, test_loader, context: dict):

    logger = context['logger']
    config = context['config']
    device = context['device']
    checkpoint: CheckPoint = context['checkpoint']


    logger.summary(config, device, train_loader, test_loader)

    num_epochs = config["epoch"]
 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config['weight_decay']
        )
   
    for epoch in range(num_epochs):

        #Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        # Test
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
        
        test_loss /= len(test_loader)
   

        # Reporting
        if epoch % 100 == 0:
            index = 0
            offset = 10
            inputs, actual = next(iter(test_loader))
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                predicted = model(inputs)

            logger.report_predict(
                epoch, 
                actual[index, 0, :].detach().cpu().numpy(), 
                predicted[index, 0, :].detach().cpu().numpy(),
                offset)
            
            checkpoint.save(epoch=epoch, model=model, optimizer=optimizer, loss=test_loss)

        logger.report_loss(epoch, train_loss, test_loss, config)
        
        
              
    