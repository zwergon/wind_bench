import torchaudio
from virtual.model import LitModel
from wind_bench.lightning_module import DataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

if __name__=="__main__":

    config = {
        'input_channels': 8,
        'output_channels': 6,
        'learning_rate': 1e-4,
        'kernel_size': 3,
        'epoch': 3,
        'batch_size': 2,
        'num_workers': 2
    }
    
    logger = TensorBoardLogger("tb-logger", name="my_model")

    dm = DataModule(config)
    model = LitModel(config)
    trainer = pl.Trainer(logger=logger, accelerator="cuda", devices=1, min_epochs=1, max_epochs=config["epoch"])
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    #trainer.test(model, dm)
