
import matplotlib.pyplot as plt

from wb.dataset import WBDataset

class Predictions:

    def __init__(self, loader, device ) -> None:
       
        self.dataset : WBDataset = loader.dataset
        self.X_torch, Y_torch = next(iter(loader))
        self.X_torch = self.X_torch.to(device)
        self.Y = Y_torch.cpu().numpy()
        self.predictions = []

    def compute(self, epoch, model):
        Y_hat_torch = model(self.X_torch)
        
        Y_hat = Y_hat_torch.detach().cpu().numpy()

        self.predictions = []

        for i in range(self.dataset.output_size):
            self.predictions.append( 
                {
                    "file": f"results_{i}_{epoch}.png", 
                    "predicted": Y_hat[0, i, :],
                    "actual": self.Y[0, i, :],
                    "y_label":self.dataset.output_name(i)
                }
            )
    
  