import torch
import torch.optim as optim

from wbvirtual.dataset import WBDataset
from wbvirtual.train.loss_function import loss_function
from wbvirtual.train.context import Context
from wbvirtual.utils.config import Config
from wbvirtual.train.predictions import Predictions
from wbvirtual.train.metrics_collection import MetricsCollection


def optimizer_function(context: Context):
    config: Config = context.config
    model = context.model
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise Exception("Unknown optimizer")

    context.checkpoint.optimizer = optimizer
    return optimizer


def scheduler_function(optimizer):
    def lr_lambda(epoch):
        # LR to be 0.1 * (1/1+0.01*epoch)
        return 0.995**epoch

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def find_lr(context: Context, train_loader):
    device = context.device
    model = context.model
    config: Config = context.config

    num_epochs = config["epochs"]

    num_batch = len(train_loader)

    optimizer = optimizer_function(context)

    loss_fct = loss_function(config, device)

    lr = config["learning_rate"]
    mult = (lr / 1e-8) ** (1 / ((num_batch * num_epochs) - 1))
    optimizer_arg = optimizer
    optimizer_arg.param_groups[0]["lr"] = 1e-8
    scheduler = optim.lr_scheduler.StepLR(optimizer_arg, step_size=1, gamma=mult)

    lrs, losses = [], []
    for epoch in range(num_epochs):
        # Train
        model.train()
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            Y_hat = model(X)

            loss = loss_fct(Y_hat, Y)
            loss.backward()
            optimizer.step()

            lr = scheduler.get_last_lr()[0]

            lrs.append(lr)
            losses.append(loss.item())

            # scheduler update
            scheduler.step()

            context.report_loss(epoch, train_loss=loss.item(), lr=lr, step=None)

    context.report_lr_find(lrs, losses)


def train_test(context: Context, train_loader, test_loader):
    device = context.device
    model = context.model
    config: Config = context.config
    train_dataset: WBDataset = train_loader.dataset

    num_epochs = config["epochs"]

    context.summary()

    criterion = loss_function(config, device)

    optimizer = optimizer_function(context)

    scheduler = scheduler_function(optimizer)

    predictions = Predictions(test_loader)

    for epoch in range(num_epochs):
        # Train
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

        if scheduler:
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
        context.report_loss(
            epoch, train_loss, test_loss=test_loss, lr=get_lr(optimizer)
        )

        context.report_metrics(epoch, train_metrics)
        context.report_metrics(epoch, test_metrics)

        if epoch % 10 == 0:
            predictions.compute(model, device=device)

            context.report_prediction(epoch, predictions)

            context.save_checkpoint(epoch=epoch, loss=test_loss)

    context.save_model()
