import json
import os


class Args:

    def __init__(self, jsonname) -> None:
        with open(jsonname, 'r') as file:
            self.__dict__ = json.load(file)

    @property
    def data_dir(self):
        return os.path.join(self.root_path, f"{self.n_samples}_{self.sequence_length}")
    
    @property
    def name(self):
        return f"{os.path.basename(self.root_path)}_{self.type.lower()}_{self.n_samples}_{self.sequence_length}"



if __name__ == "__main__":
    
    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))
    print(args.learning_rate)
    print(args.type)
    print(args.data_dir)
    print(args.name)