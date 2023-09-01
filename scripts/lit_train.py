import os
import torchaudio
from wind_bench.data_module import WBDataModule
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from virtual.models.model import LitModel

from args import Args

if __name__=="__main__":

    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))
    
    logger = TensorBoardLogger("tb-logger", name="my_model")

    dm = WBDataModule(args.__dict__)

    dm.setup("test") 
    test_dataset = dm.test_dataset

    model = LitModel(test_dataset.input_size, test_dataset.output_size, args.__dict__)
    trainer = pl.Trainer(logger=logger, accelerator="cuda", devices=1, min_epochs=1, max_epochs=args.epoch)
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    # trainer.test(model, dm)
