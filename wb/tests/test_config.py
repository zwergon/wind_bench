import unittest
import os

from wb.utils.config import Config


class TestConfig(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_config(self):
        config = Config.test_config()
        self.assertTrue(config is not None)
        self.assertListEqual(config.indices, [3])

        config = Config.toy_config()
        self.assertTrue(config is not None)
        self.assertEqual(config.sensors, "toy")

        config = Config.default_config()
        self.assertTrue(config is not None)
        self.assertEqual(config.sensors, "wind_bench")

    def test_config_file(self):
        config = Config(
            os.path.join(os.path.dirname(__file__), "../data", "config.json")
        )
        self.assertTrue(config is not None)
        self.assertEqual(config.project, "wb_virtual_tests")


if __name__ == "__main__":
    unittest.main()
