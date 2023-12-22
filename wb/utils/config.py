import json
import os


from enum import IntEnum


class FSType(IntEnum):
    FILE = 0
    AZURE = 1
    S3 = 2


class FileType(IntEnum):
    NUMPY = 0
    PARQUET = 1


class Config:
    @staticmethod
    def create_from_args(args):
        config = Config(args.config)
        config.__dict__.update(args.__dict__)
        return config

    def __init__(self, config_name) -> None:
        with open(config_name, "r") as file:
            self.__dict__ = json.load(file)
        self.dirname = os.path.dirname(config_name)
