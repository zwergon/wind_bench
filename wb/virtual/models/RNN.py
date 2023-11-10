import torch
import torch.nn as nn



# The Neural Network
# according to https://www.kaggle.com/code/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99
class RNNVanilla(nn.Module):
    # __init__: the function where we create the architecture
    def __init__(self, input_size, output_size, hidden_size, dropout):
        super(RNNVanilla, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, self.hidden_size, batch_first=True)

        # Fully Connected Layer
        self.layer = nn.Linear(self.hidden_size, output_size)
        
        #dropout
        self.drop = nn.Dropout(dropout)
    # forward: function where we apply the architecture to the input
    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.drop(out)
        out = self.layer(out)
        out = torch.swapaxes(out, 1, 2)
        return out 