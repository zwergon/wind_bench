import os
import torch


class CheckPoint:
    def __init__(self, path=None) -> None:
        self.params = {}
        self.model = None
        self.optimizer = None
        self.signature = None
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

    def save(self, filename, epoch, loss):
        assert (
            self.model is not None
        ), "model need first to be assigned to checkpoint before saving"
        self.params["epoch"] = epoch
        self.params["loss"] = loss
        self.params["model_state_dict"] = self.model.state_dict()
        if self.optimizer is not None:
            self.params["optimizer_state_dict"] = self.optimizer.state_dict()
        torch.save(self.params, filename)
