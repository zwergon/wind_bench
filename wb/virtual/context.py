import os
import torch
import uuid
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.schema import TensorSpec, Schema

from wb.utils.config import Config
from wb.utils.display import predictions_plot
from wb.dataset import WBDataset
from wb.virtual.checkpoint import CheckPoint
from wb.virtual.predictions import Predictions
from wb.virtual.models import get_model


class Context:
    @staticmethod
    def _experiment_name(root_name):
        return f"{root_name}_{str(uuid.uuid1())[:8]}"

    def __init__(self, config: Config, checkpoint=CheckPoint()) -> None:
        self._config = config
        self.experiment_id = None
        self.checkpoint: CheckPoint = checkpoint

        self.device = torch.device(
            "cuda" if config.cuda and torch.cuda.is_available() else "cpu"
        )

    def __enter__(self):
        if self.config["tracking_uri"]:
            mlflow.set_tracking_uri(self.config["tracking_uri"])
            self.experiment_id = mlflow.create_experiment(
                self._experiment_name(self.config["project"])
            )
        matplotlib.use("agg")
        mlflow.start_run(experiment_id=self.experiment_id)
        mlflow.log_params(self.config)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @property
    def config(self):
        return self._config.__dict__

    @property
    def model(self):
        return self.checkpoint.model

    def close(self):
        matplotlib.use("TkAgg")
        mlflow.end_run()

    def create_model(self, loader):
        dataset: WBDataset = loader.dataset
        model = get_model(
            self.config, dataset.input_size, dataset.output_size, self.device
        )
        self.checkpoint.model = model

        X, Y = next(iter(loader))

        input_schema = Schema(
            [TensorSpec(type=np.dtype(np.float32), shape=(-1, X.shape[1], X.shape[2]))]
        )
        output_schema = Schema(
            [TensorSpec(type=np.dtype(np.float32), shape=(-1, Y.shape[1], Y.shape[2]))]
        )
        self.checkpoint.signature = ModelSignature(
            inputs=input_schema, outputs=output_schema
        )

        return model

    def __str__(self):
        msg = f"Type Network: {self._config.type}\n"
        msg += f"Device : {self.device}\n"
        msg += f"matplotlib backend: {matplotlib.rcParams['backend']}, interactive: {matplotlib.is_interactive()}\n"
        active_run = mlflow.active_run()
        if active_run:
            msg += f"Name: {active_run.info.run_name}\n"
            msg += f"Experiment_id: {active_run.info.experiment_id}\n"
            msg += f"Artifact Location: {active_run.info.artifact_uri}\n"
        if self.checkpoint.signature is not None:
            msg += f"Signature {str(self.checkpoint.signature)}\n"

        return msg

    def summary(self):
        print(str(self))
        mlflow.log_text(str(self), "summary.txt")

    def save_checkpoint(self, epoch, loss):
        with tempfile.TemporaryDirectory() as tmpdirname:
            ckp_name = os.path.join(tmpdirname, f"checkpoint_{epoch}_{loss:.2f}.pth")
            self.checkpoint.save(ckp_name, epoch=epoch, loss=loss)

            mlflow.log_artifact(local_path=ckp_name, artifact_path="checkpoints")

    def save_model(self):
        model = self.checkpoint.model
        signature = self.checkpoint.signature
        mlflow.pytorch.log_model(model, "model", signature=signature)

    def report_loss(self, epoch, train_loss, lr, test_loss=None, step=10):
        num_epochs = self.config["epochs"]

        losses = {"train_loss": train_loss, "lr": lr}
        if test_loss is not None:
            losses["test_loss"] = test_loss
        mlflow.log_metrics(losses, step=epoch)

        if step is None or epoch % step == 0:
            if test_loss is None:
                print(
                    f"Epoch {epoch}/{num_epochs} - Loss: train {train_loss:.6f}, lr {lr:.2e}"
                )
            else:
                print(
                    f"Epoch {epoch}/{num_epochs} - Loss: train {train_loss:.6f}, test {test_loss:.6f}, lr {lr:.2e}"
                )

    def report_metrics(self, epoch, metrics):
        values = {}
        for k, v in metrics.results.items():
            if len(v.shape) == 0:
                values[k] = v.item()
            else:
                for c in range(v.shape[0]):
                    values[f"{k}_{c}"] = v[c].item()
        mlflow.log_metrics(values, step=epoch)

    def report_prediction(self, epoch, predictions: Predictions, index=0):
        fig = predictions_plot(predictions=predictions)
        mlflow.log_figure(fig, f"actual_predicted_{epoch}.png")
        plt.close(fig)

    def report_lr_find(self, lr, losses):
        from wb.utils.display import lrfind_plot

        fig = lrfind_plot(lr, losses)
        mlflow.log_figure(fig, "lr_find.png")
        plt.close(fig)
