
from clearml import Task, Dataset
import matplotlib.pyplot as plt
import numpy as np


class Logger:

    def __init__(self) -> None:
        self.logger = None
        self.clearml = True

    def init(self, task_name, config):
        print(f"init {task_name}")
        if self.clearml:
            task = Task.init(project_name=config['project'], task_name=task_name)
            task.connect(config)
            self.logger = task.get_logger()


    def summary(self, config, device, train_loader, test_loader):
        print(f"Device : {device}")
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        print(f"X_train : {X_train.shape}")
        print(f"y_train : {y_train.shape}")
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}")
        print(f"Type Network: {config['type']}")

    def report_loss(self, epoch, train_loss, test_loss, mae_train, mae_test, config):
        num_epochs = config['epoch']

        if self.logger is not None:
            self.logger.report_scalar("Train/Test dilate", "Train", train_loss, iteration=epoch)
            self.logger.report_scalar("Train/Test dilate", "Test", test_loss, iteration=epoch)
            self.logger.report_scalar("Train/Test mae", "Train", mae_train, iteration=epoch)
            self.logger.report_scalar("Train/Test mae", "Test", mae_test, iteration=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, MAE train: {mae_train:.4f}, MAE test: {mae_test:.4f}")

        
    def report_predict(self, epoch, real_outputs, predicted_outputs, offset=0):

        if self.logger is not None:
            scatter2d = [(i, real_outputs[i]) for i in range(offset, len(real_outputs))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Mudline Moment"
            )

            scatter2d = [(i, predicted_outputs[i]) for i in range(offset, len(real_outputs))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Mudline Moment"
            )
