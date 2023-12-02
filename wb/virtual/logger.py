
import matplotlib.pyplot as plt
import numpy as np

import os
import mlflow



class Logger:

    def __init__(self, config: dict) -> None:
        self.logger = None
        self.config = config
        path =  "/tmp/mlrun"
        mlflow.set_tracking_uri(f"file://{path}")
        experiment_id = mlflow.create_experiment(
            "Social NLP Experiments",
            artifact_location=path
            )
        experiment = mlflow.get_experiment(experiment_id)
        print(f"Name: {experiment.name}")
        print(f"Experiment_id: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        
        mlflow.start_run(experiment_id=experiment.experiment_id)

    def close(self):
        mlflow.end_run()
        
    def summary(self, device, train_loader, test_loader):
        print(f"Device : {device}")
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        print(f"X_train : {X_train.shape}")
        print(f"y_train : {y_train.shape}")
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}") 
        print(f"Type Network: {self.config['type']}")

        

    def report_loss(self, epoch, losses:dict):
        num_epochs = self.config['epoch']

        for loss in losses.keys():
            mlflow.log_metrics( {f"train_{loss}": losses[loss][0], f"test_{loss}":losses[loss][1]}, step=epoch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}")
            for loss in losses.keys():
                print(f"train_{loss}: {losses[loss][0]}, test_{loss}: {losses[loss][1]}")
        
    # def report_predict(self, epoch, mud_x, pred_mud_x, mud_y, pred_mud_y, mud_z, pred_mud_z, 
    #                    wat_x, pred_wat_x, wat_y, pred_wat_y, wat_z, pred_wat_z, offset=0):

    #     if self.logger is not None:
    #         scatter2d = [(i, mud_x[i]) for i in range(offset, len(mud_x))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment_Mx {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Mudline Moment Mx"
    #         )

    #         scatter2d = [(i, pred_mud_x[i]) for i in range(offset, len(mud_x))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment_Mx {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Mudline Moment Mx"
    #         )

    #         scatter2d = [(i, mud_y[i]) for i in range(offset, len(mud_y))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment_My {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="mudline Moment My"
    #         )

    #         scatter2d = [(i, pred_mud_y[i]) for i in range(offset, len(mud_y))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment_My {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="mudline Moment My"
    #         )

    #         scatter2d = [(i, mud_z[i]) for i in range(offset, len(mud_z))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment Mz {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="mudline Moment Mz"
    #         )

    #         scatter2d = [(i, pred_mud_z[i]) for i in range(offset, len(mud_z))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Mudline moment Mz {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="mudline Moment Mz"
    #         )

    #         scatter2d = [(i, wat_x[i]) for i in range(offset, len(wat_x))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline Moment Mx {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment Mx"
    #         )

    #         scatter2d = [(i, pred_wat_x[i]) for i in range(offset, len(wat_x))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline Moment Mx {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment Mx"
    #         )

    #         scatter2d = [(i, wat_y[i]) for i in range(offset, len(wat_y))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline Moment My {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment My"
    #         )

    #         scatter2d = [(i, pred_wat_y[i]) for i in range(offset, len(wat_y))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline Moment My {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment My"
    #         )

    #         scatter2d = [(i, wat_z[i]) for i in range(offset, len(wat_z))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline moment Mz {epoch}',
    #             "Actual",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment Mz"
    #         )

    #         scatter2d = [(i, pred_wat_z[i]) for i in range(offset, len(wat_z))]
    #         self.logger.report_scatter2d(
    #             f'Actual vs. Predicted Waterline moment Mz {epoch}',
    #             "Predicted",
    #             iteration=epoch,
    #             scatter=scatter2d,
    #             xaxis="Time",
    #             yaxis="Waterline moment Mz"
    #         )
