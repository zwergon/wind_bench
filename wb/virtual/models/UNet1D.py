import torch
import torch.nn as nn
import torch.optim as optim

class UNet1D(nn.Module):
    def __init__(self, input_size, output_size):
        super(UNet1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, output_size, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Adjust activation function based on your data
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Transpose input to match channel dimension
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)  # Transpose back to original shape
        return decoded
