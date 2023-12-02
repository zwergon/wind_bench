import os
import torch

class CheckPoint:

    def __init__(self, root_path, kind) -> None:
        self.path = root_path
        self.params = {
            "kind": kind
        }

    @property
    def state_dict(self):
        return self.params['model_state_dict']


    @staticmethod
    def load(checkpoint_path: str):
        params = torch.load(checkpoint_path)
        checkpoint = CheckPoint(os.path.basename(checkpoint_path), params['kind'])
        checkpoint.params.update(params)

        return checkpoint

    def save(self, epoch, model, optimizer, loss):
        
        filename = os.path.join(self.path, f"checkpoint_{self.kind}_{epoch}_{loss:.2f}.pth")
        self.params['epoch'] = epoch
        self.params['model_state_dict'] = model.state_dict()
        self.params['optimizer_state_dict'] = optimizer.state_dict()
        self.params['loss'] = loss
        torch.save(self.params, filename)

