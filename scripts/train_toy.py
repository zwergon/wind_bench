import torchaudio
from model import LitModel
from dataset import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__=="__main__":

    config = {
        'input_channels': 4,
        'output_channels': 1,
        'learning_rate': 1e-4,
        'kernel_size': 3,
        'epoch': 1,
        'batch_size': 204,
        'num_workers': 4
    }
    
    logger = TensorBoardLogger("tb-logger", name="my_model")

    dm = DataModule(config)
    model = LitModel(config)
    trainer = pl.Trainer(logger=logger, accelerator="cuda", devices=1, min_epochs=1, max_epochs=config["epoch"])
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
