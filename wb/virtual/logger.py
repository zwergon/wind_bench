
from clearml import Task, Dataset
import matplotlib.pyplot as plt
import numpy as np

import mlflow


class Logger:

    def __init__(self) -> None:
        self.logger = None
        self.clearml = False

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

        

    def report_loss(self, epoch, train_loss, test_loss, mae_train, mae_test, config, lr, r2_train, r2_test):
        num_epochs = config['epoch']

        if self.logger is not None:
            self.logger.report_scalar("Train/Test mse", "Train", train_loss, iteration=epoch)
            self.logger.report_scalar("Train/Test mse", "Test", test_loss, iteration=epoch)
            self.logger.report_scalar("Train/Test mae", "Train", mae_train, iteration=epoch)
            self.logger.report_scalar("Train/Test mae", "Test", mae_test, iteration=epoch)
            self.logger.report_scalar("learning_rate", "Lr", lr, iteration=epoch)
            self.logger.report_scalar("Train/Test R2", "Train", r2_train, iteration=epoch)
            self.logger.report_scalar("Train/Test R2", "Test", r2_test, iteration=epoch)

        mlflow.log_metrics( {"train_loss": train_loss, "test_loss":test_loss}, step=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, MAE train: {mae_train:.4f}, MAE test: {mae_test:.4f}, R2 train: {r2_train:.4f}, R2 test: {r2_test:.4f}")
            #print(f"lr_before {before_lr} lr_after {after_lr}")
        
    def report_predict(self, epoch, mud_x, pred_mud_x, mud_y, pred_mud_y, mud_z, pred_mud_z, 
                       wat_x, pred_wat_x, wat_y, pred_wat_y, wat_z, pred_wat_z, offset=0):

        if self.logger is not None:
            scatter2d = [(i, mud_x[i]) for i in range(offset, len(mud_x))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment_Mx {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Mudline Moment Mx"
            )

            scatter2d = [(i, pred_mud_x[i]) for i in range(offset, len(mud_x))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment_Mx {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Mudline Moment Mx"
            )

            scatter2d = [(i, mud_y[i]) for i in range(offset, len(mud_y))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment_My {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="mudline Moment My"
            )

            scatter2d = [(i, pred_mud_y[i]) for i in range(offset, len(mud_y))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment_My {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="mudline Moment My"
            )

            scatter2d = [(i, mud_z[i]) for i in range(offset, len(mud_z))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment Mz {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="mudline Moment Mz"
            )

            scatter2d = [(i, pred_mud_z[i]) for i in range(offset, len(mud_z))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Mudline moment Mz {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="mudline Moment Mz"
            )

            scatter2d = [(i, wat_x[i]) for i in range(offset, len(wat_x))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline Moment Mx {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment Mx"
            )

            scatter2d = [(i, pred_wat_x[i]) for i in range(offset, len(wat_x))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline Moment Mx {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment Mx"
            )

            scatter2d = [(i, wat_y[i]) for i in range(offset, len(wat_y))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline Moment My {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment My"
            )

            scatter2d = [(i, pred_wat_y[i]) for i in range(offset, len(wat_y))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline Moment My {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment My"
            )

            scatter2d = [(i, wat_z[i]) for i in range(offset, len(wat_z))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline moment Mz {epoch}',
                "Actual",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment Mz"
            )

            scatter2d = [(i, pred_wat_z[i]) for i in range(offset, len(wat_z))]
            self.logger.report_scatter2d(
                f'Actual vs. Predicted Waterline moment Mz {epoch}',
                "Predicted",
                iteration=epoch,
                scatter=scatter2d,
                xaxis="Time",
                yaxis="Waterline moment Mz"
            )
