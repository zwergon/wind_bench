

import os
import torch
import uuid
import mlflow
from mlflow.models import infer_signature
import tempfile


import matplotlib.pyplot as plt
import numpy as np

from wb.utils.config import Config
from wb.virtual.checkpoint import CheckPoint
from wb.virtual.predictions import Predictions


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
        mlflow.log_params(self.config)
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


    def report_loss(self, epoch, train_loss, test_loss, lr):
        num_epochs = self.config['epochs']

        mlflow.log_metrics( 
                    {
                        "train_loss": train_loss, 
                        "test_loss": test_loss,
                        "lr": lr
                    }, 
                    step=epoch
                )

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs} - Loss: train {train_loss:.6f}, test {test_loss:.6f}, lr {lr:.6f}")

    def report_metrics(self, epoch, metrics):
        values = {}
        for k, v in metrics.results.items():
            if len(v.shape) == 0:
                values[k] = v.item()
            else:
                for c in range(v.shape[0]):
                    values[f"{k}_{c}"] = v[c].item()
        mlflow.log_metrics( values, step=epoch)
      

    def report_prediction(self, predictions: Predictions):
        for data in predictions.predictions:
            fig, ax = plt.subplots()
            ax.plot(data['predicted'])
            ax.plot(data['actual'])
            ax.set_ylabel(data['y_label'])
            mlflow.log_figure(fig, data['file'])
            plt.close(fig)
            
  