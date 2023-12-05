

import os
import torch
import uuid
import mlflow
import tempfile


import matplotlib.pyplot as plt
import numpy as np

from wb.utils.config import Config
from wb.virtual.checkpoint import CheckPoint



class Context:

    @staticmethod
    def _experiment_name(root_name):
        return f"{root_name}_{str(uuid.uuid1())[:8]}"

    def __init__(self, config: Config) -> None:
        self._config = config

        self.checkpoint = CheckPoint()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)
            self.experiment_id = mlflow.create_experiment(self._experiment_name(config.project))
        else:
            self.experiment_id = None

    def __enter__(self):
        mlflow.start_run(experiment_id=self.experiment_id)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def config(self):
        return self._config.__dict__

    def close(self):
        mlflow.end_run()
        
    def summary(self, train_loader, test_loader):
        print(f"Device : {self.device}")
        active_run = mlflow.active_run()
        if active_run:
            print(f"Name: {active_run.info.run_name}")
            print(f"Experiment_id: {active_run.info.experiment_id}")
            print(f"Artifact Location: {active_run.info.artifact_uri}")
        print(f"Type Network: {self._config.type}")
        X_train, y_train = next(iter(train_loader))
        X_test, y_test = next(iter(test_loader))
        print(f"X_train : {X_train.shape}")
        print(f"y_train : {y_train.shape}")
        print(f"X_test : {X_test.shape}")
        print(f"y_test : {y_test.shape}") 


    def save_checkpoint(self, epoch, model, optimizer, loss):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckp_name = os.path.join(tmpdirname, f"checkpoint_{epoch}_{loss:.2f}.pth")
            self.checkpoint.save(
                            ckp_name,
                            epoch=epoch, 
                            model=model, 
                            optimizer=optimizer, 
                            loss=loss
                            )
        
            mlflow.log_artifact(local_path=ckp_name, artifact_path="checkpoints")


    def report_loss(self, epoch, train_loss, test_loss):
        num_epochs = self.config['epoch']

        mlflow.log_metrics( {"train_loss": train_loss, "test_loss": test_loss}, step=epoch)

        if epoch % 10 == 0:
            print(f"Loss - train: {train_loss:.6f}, test: {test_loss:.6f} Epoch {epoch}/{num_epochs}")

    def report_metrics(self, epoch, train_metrics, test_metrics):
        train_metrics_one = {k:v.detach().cpu().numpy()[0] for k, v in train_metrics.items()}
        mlflow.log_metrics( train_metrics_one, step=epoch)
        # mlflow.log_metrics( test_metrics.results, step=epoch)

      

    def report_prediction(self, predictions: list):
        for p in predictions:
            fig = p.plot()
            mlflow.log_figure(fig, p.file)
            plt.close(fig)
        
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
