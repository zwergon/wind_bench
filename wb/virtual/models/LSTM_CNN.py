import torch
import torch.nn as nn


# Step 4: Define the LSTM model
class LSTMCNNModel(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(LSTMCNNModel, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]

        self.encodeur = nn.Sequential(
            nn.Conv1d(input_size, 7, kernel_size=3, padding="same"),
            nn.BatchNorm1d(7),
            nn.ReLU(),
            nn.Conv1d(7, 5, kernel_size=3, padding="same"),
        )

        padding = (3 - 1) // 2
        self.decodeur = nn.Sequential(
            nn.ConvTranspose1d(5, 7, kernel_size=3, padding=padding),
            nn.BatchNorm1d(7),
            nn.ReLU(),
            nn.ConvTranspose1d(7, 6, kernel_size=3, padding=padding),
        )

        self.lstm = nn.LSTM(6, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.encodeur(x)
        x = self.decodeur(x)
        x = torch.swapaxes(x, 1, 2)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, (h0, c0) = self.lstm(x, (h0, c0))
        # out = self.sigmoid(out)
        out = self.fc(out)
        # out = self.sigmoid(out)
        out = torch.swapaxes(out, 1, 2)
        return out
