import unittest
import os

from wb.utils.config import Config


class TestConfig(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_config(self):
        config = Config(os.path.join(os.path.dirname(__file__), "config.json"))
        self.assertTrue(config is not None)


if __name__ == "__main__":
    unittest.main()
