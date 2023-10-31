import torch
from torch import nn
import torch.optim as optim
from virtual.checkpoint import CheckPoint
from virtual.loss_function import get_loss


def train_test(model, train_loader, test_loader, context: dict):

    logger = context['logger']
    config = context['config']
    device = context['device']
    checkpoint: CheckPoint = context['checkpoint']
   

    logger.summary(config, device, train_loader, test_loader)

    num_epochs = config["epoch"]

    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        weight_decay=config['weight_decay']
        )
    
    labels_list_train = []
    outputs_list_train = []

    labels_list_test = []
    outputs_list_test = []

    for epoch in range(num_epochs):

        #Train
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = get_loss(device, config)(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            labels_list_train.append(labels.cpu())
            outputs_list_train.append(outputs.detach().cpu())

        train_loss /= len(train_loader)

        all_labels = torch.cat(labels_list_train, dim=0)
        all_outputs= torch.cat(outputs_list_train, dim=0)
            
        mae_train =  nn.L1Loss()(all_labels, all_outputs)

        # Test
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                
                #loss, loss_shape, loss_temporal = dilate_loss(outputs,labels,0.3,0.005, device)
                loss = get_loss(device, config)(outputs, labels)
                test_loss += loss.item()
                
                labels_list_test.append(labels.cpu())
                outputs_list_test.append(outputs.cpu())
        test_loss /= len(test_loader)
        
        all_labels_test = torch.cat(labels_list_test, dim=0)
        all_outputs_test= torch.cat(outputs_list_test, dim=0)
            
        mae_test =  nn.L1Loss()(all_labels_test, all_outputs_test)

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

        logger.report_loss(epoch, train_loss, test_loss, mae_train, mae_test, config)
        
        
              
    