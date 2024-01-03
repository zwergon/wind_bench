import unittest

from wbvirtual.dataset.sensor_description import SensorDescr


class TestSensorDescr(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_wb_sensor(self):
        sensor_descr = SensorDescr.create("wind_bench")
        self.assertListEqual(
            sensor_descr.y_columns,
            [
                "Mudline moment Mx[kNm]",
                "Mudline moment My[kNm]",
                "Mudline moment Mz[kNm]",
                "Waterline moment Mx[kNm]",
                "Waterline moment My[kNm]",
                "Waterline moment Mz[kNm]",
            ],
        )

        sensor2 = SensorDescr.create("wind_bench")
        self.assertEqual(sensor_descr, sensor2)

    def test_toy_sensor(self):
        sensor_descr = SensorDescr.create("toy")
        self.assertListEqual(sensor_descr.y_columns, ["dof3"])

        sensor2 = SensorDescr.create("toy")
        self.assertEqual(sensor_descr, sensor2)


if __name__ == "__main__":
    unittest.main()
