
import json
import os

from wb.dataset.utils.meta_singleton import MetaSingleton


class Config(object, metaclass=MetaSingleton):
    CONFIG = {
        "protocol": "http",
        "s3": {
            "url": "minio.10.68.0.250.nip.io:80",
            "acces_key": "minio",
            "secret_key": "minio_key",
            "bucket": "mpu"
        }
    }
    compteur = 0

  
    @staticmethod
    def load_from_dict(dict):
        Config.CONFIG.update(dict)
        
    
    @staticmethod
    def load_from_file(ini_file=None):
        Config.compteur += 1
        if ini_file is not None:
            with open(ini_file, "r") as fp:
                dict = json.loads(fp.read())
                Config.load_from_dict(dict)

    @property
    def PROTOCOL(self):
        return self.CONFIG['protocol']

    @property
    def S3_URL(self):
        return f"{self.CONFIG['protocol']}://{self.CONFIG['s3']['url']}"
    
    @property
    def S3_ACCESS_KEY(self):
        return self.CONFIG["s3"]["acces_key"]
    
    @property
    def S3_SECRET_KEY(self):
        return self.CONFIG["s3"]["secret_key"]
    
    @property
    def S3_BUCKET(self):
        return self.CONFIG["s3"]["bucket"]
    
    
  