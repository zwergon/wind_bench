
import unittest
import os

from wb.utils.args import Args

class TestArgs(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        

    def test_args(self): 
        args = Args(os.path.join(os.path.dirname(__file__), "args.json"))
        print(args.learning_rate)
        print(args.type)
        print(args.data_dir)
        print(args.name)
        self.assertTrue(args is not None)

if __name__ == "__main__":
    unittest.main()