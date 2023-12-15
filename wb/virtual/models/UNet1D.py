import torch
import torch.nn as nn

class UNet1D(nn.Module):
    def __init__(self, input_size, output_size, kernel):
        super(UNet1D, self).__init__()

        self.conv_bloc_1 = nn.Sequential(
            nn.Conv1d(input_size, 8, kernel_size=kernel, padding='same'),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=kernel, padding="same"),
            nn.ReLU()
        )

        self.conv_bloc_2 = nn.Sequential(
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=kernel, padding="same"),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=kernel, padding="same"),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool1d(2)
        self.sampling = nn.Upsample(scale_factor=2)
        self.wg_1 = nn.Conv1d(16, 16, kernel_size=kernel, padding="same")
        self.ws_1 = nn.Conv1d(16, 16, kernel_size=kernel, padding="same")

        self.dec_block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=kernel, padding="same"),
            nn.Sigmoid()
        )
        
        self.wg_2 =  nn.Conv1d(16, 8, kernel_size=kernel, padding="same")
        self.ws_2 = nn.Conv1d(8, 8, kernel_size=kernel, padding="same")
        
        self.dec_block_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=kernel, padding="same"),
            nn.Sigmoid()
        )

        self.conv_out = nn.Conv1d(8, output_size, kernel_size=kernel, padding="same")
        self.fc = nn.Linear(output_size, output_size)

    def forward(self, x):
        half_lay = []

        x = self.conv_bloc_1(x)
        half_lay.append(x)
        
        x = self.conv_bloc_2(x)
        half_lay.append(x)
       
        x = self.maxpool(x)
        x = self.sampling(x)

        wg = self.wg_1(x)
        ws = self.ws_1(half_lay[1])

        fact = self.dec_block_1(wg+ws)

        x = fact*half_lay[1]
        x = self.sampling(x)
        
        wg = self.wg_2(x)
        ws = self.ws_2(half_lay[0])
    
        fact = self.dec_block_2(wg+ws)

        x = fact*half_lay[0] 
        x = self.conv_out(x)

        x = self.fc(torch.swapaxes(x, 1, 2))
        x = torch.swapaxes(x, 1, 2)
        return x         
