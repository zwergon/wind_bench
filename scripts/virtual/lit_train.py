import os
import torchaudio
from wb.dataset.data_module import WBDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from wb.virtual.models.model import LitModel

from wb.utils.config import Config

if __name__=="__main__":

    config = Config(jsonname = os.path.join(os.path.dirname(__file__), "config.json"))
    
    logger = TensorBoardLogger("tb-logger", name="my_model")

    dm = WBDataModule(config.__dict__)

    dm.setup("test") 
    test_dataset = dm.test_dataset

    model = LitModel(test_dataset.input_size, test_dataset.output_size, config.__dict__)
    trainer = pl.Trainer(logger=logger, accelerator="cuda", devices=1, min_epochs=1, max_epochs=config.epoch)
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    # trainer.test(model, dm)
