import torch
import torch.nn as nn


# Step 4: Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(LSTMModel, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            bias=False,
            dropout=0.1,
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)

        h0 = torch.ones(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.ones(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, (h0, c0) = self.lstm(x, (h0, c0))
        out = self.fc(out)
        out = torch.swapaxes(out, 1, 2)
        return out
