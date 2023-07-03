import torch.nn as nn
import numpy as np
import torch
from ai.dataset import FSBDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss

device = torch.device("cuda")

all_dataset=FSBDataset("D:/work/ai.virtual/data/five_story_building_ts_with_us_1000.npy")
dataset = torch.utils.data.Subset(all_dataset, range(48000) )

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=204, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=204, shuffle=False)

class Conv1dAutoEncodeur(nn.Module):
    def __init__(self):
        super(Conv1dAutoEncodeur, self).__init__()
        self.encodeur=nn.Sequential(
                      nn.Conv1d(4, 32, kernel_size=3),
                      nn.ReLU(),
                      nn.Conv1d(32, 16, kernel_size=3),
                      nn.ReLU(),
                      nn.Conv1d(16, 8, kernel_size=3) 
                            )
        self.decodeur=nn.Sequential(
                      nn.ConvTranspose1d(8, 16, kernel_size=3),
                      nn.ReLU(),
                      nn.ConvTranspose1d(16, 32, kernel_size=3),
                      nn.ReLU(),
                      nn.ConvTranspose1d(32, 1, kernel_size=3)        
        )

    def forward(self, x): 
        x = self.encodeur(x)
        x = self.decodeur(x)
        x = x.squeeze(dim=1)
        return x

model = Conv1dAutoEncodeur().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

trainer= create_supervised_trainer(model, optimizer, criterion, device)

train_loss=[]
test_loss=[]

val_metrics={
            "loss": Loss(criterion)
            }

train_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    train_loss.append(metrics['loss'])
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.4f}")


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    val_evaluator.run(test_loader)
    metrics = val_evaluator.state.metrics
    test_loss.append(metrics['loss'])
    print(f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.4f}")

trainer.run(train_loader, max_epochs=100)

plt.figure()
plt.plot(np.arange(100), train_loss, label="loss_train")
plt.plot(np.arange(100), test_loss, label="loss_test")
plt.legend()
plt.show()