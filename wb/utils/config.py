import json
import os

 
from enum import IntEnum

class FSType(IntEnum):
    FILE=0
    AZURE=1
    S3=2

class FileType(IntEnum):
    NUMPY=0
    PARQUET=1


class Config:

    def __init__(self, args) -> None:
        with open(args.config, 'r') as file:
            self.__dict__ = json.load(file)
        self.dirname = os.path.dirname(args.config)

        self.__dict__.update(args.__dict__)
        
    def _fstype(self):
        print(self.root_path)
        idx = self.root_path.find("file://")

        if idx >= 0:
            path = self.root_path[7:]
            if path[0] == '.':
                path = path.replace(".", self.dirname)
            fs_type = FSType.FILE
        else:
            idx = self.root_path.find("azureml://")
            if idx >= 0:
                path = self.root_path
                fs_type = FSType.AZURE
            else:
                raise Exception("filesystem not known")

        return fs_type, path

    def _filetype(self):
        if "numpy" in self.file:
            f_type = FileType.NUMPY
        else:
            f_type = FileType.PARQUET

        return f_type, self.file
    
      
    @property
    def name(self):
        return f"{os.path.basename(self.root_path)}_{self.type.lower()}_{self.n_samples}_{self.sequence_length}"

    @property
    def data_dir(self):
        fs_type, path = self._fstype()
        return fs_type, os.path.join(path, f"{self.n_samples}_{self.sequence_length}")


