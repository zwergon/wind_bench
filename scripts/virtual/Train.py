import os
import torch
from torch.utils.data import DataLoader

from wb.dataset import dataset
import ray
from ray import tune
from functools import partial

from wb.virtual.models import get_model
from wb.virtual.Training import train_test

from wb.virtual.logger import Logger
from wb.virtual.checkpoint import CheckPoint

from wb.utils.args import Args, FSType

if __name__ == "__main__":
    
    #INPUT Parameters
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))

    logger = Logger()
    logger.init(task_name=args.name, config=args.__dict__)

    checkpoint = CheckPoint(".", args.type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      #conf ray_tune
    conf={
    "learning_rate":tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(3e-5, 1e-1),
    #"alpha":0.3,
    #"gamma":0.005
    } 
  

    train_dataset, test_dataset = dataset(args)
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = get_model(
        train_dataset.input_size,  
        train_dataset.output_size,
        args.__dict__)
    model = model.to(device)

    context = {
        "config": args.__dict__,
        "device": device,
        "logger": logger,
        "checkpoint": checkpoint
    }


    
    if args.tune == True:
        ray.init(num_gpus=1)
        result = tune.run(partial(train_test, model=model, train_loader=train_loader, test_loader=test_loader, context=context),
                       config=conf, num_samples=20, metric="r2_score", mode="max",resources_per_trial={"cpu": 1, "gpu": 1})
        print(f"Best config: {result.get_best_config()}")


    train_test(conf, model, train_loader, test_loader, context)
