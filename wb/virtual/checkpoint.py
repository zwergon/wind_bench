import os
import torch

class CheckPoint:

    def __init__(self, root_path, kind) -> None:
        self.path = root_path
        self.kind = kind

    def save(self, epoch, model, optimizer, loss):
        
        filename = os.path.join(self.path, f"checkpoint_{self.kind}_{epoch}_{loss:.2f}.pth")
        torch.save({
                    'epoch': epoch,
                    'kind': self.kind, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, filename)

