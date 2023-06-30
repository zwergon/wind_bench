from ai.dataset import FSBDataset
from torch.utils.data import DataLoader
from ai_virtual.model import get_model
from ai_virtual.training import train_test_model
import torch


if __name__=="__main__":
    config={
        'model_name': "ConvAutoEnc",
        'input_channels': 4,
        'output_channels': 1,
        'learning_rate': 1e-4,
        'kernel_size': 3,
        'epoch': 100,
        'batch_size': 2048
    }

    all_dataset=FSBDataset("D:/lecomtje/Repositories/MPU/toymodel/git-toymodel/data/five_story_building_ts_with_us_1000.npy")

    dataset = torch.utils.data.Subset(all_dataset, range(48000) )


    print(f"Dataset size {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True)
    
    device = torch.device("cuda")


    train_test_model(get_model(config), train_loader, test_loader, config, device)
