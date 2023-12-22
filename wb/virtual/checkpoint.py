import os
import torch


class CheckPoint:
    def __init__(self, path=None) -> None:
        self.params = {}
        self.path = path

    @property
    def state_dict(self):
        return self.params["model_state_dict"]

    @staticmethod
    def load(checkpoint_path: str):
        params = torch.load(checkpoint_path)
        checkpoint = CheckPoint(os.path.basename(checkpoint_path))
        checkpoint.params.update(params)

        return checkpoint

    def save(self, filename, epoch, model, optimizer, loss):
        self.params["epoch"] = epoch
        self.params["loss"] = loss
        self.params["model_state_dict"] = model.state_dict()
        self.params["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(self.params, filename)
