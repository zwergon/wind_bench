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

    @staticmethod
    def default_path():
        return os.path.join(os.path.dirname(__file__), "cf", "config.json")

    @staticmethod
    def default_config():
        return Config(Config.default_path)

    @staticmethod
    def test_config():
        return Config(os.path.join(os.path.dirname(__file__), "cf", "config_test.json"))

    @staticmethod
    def toy_config():
        return Config(os.path.join(os.path.dirname(__file__), "cf", "config_toy.json"))

    def __init__(self, config_name) -> None:
        with open(config_name, "r") as file:
            self.__dict__ = json.load(file)
        self.dirname = os.path.dirname(config_name)
