from wb.utils.meta_singleton import MetaSingleton


class SensorDescr(object, metaclass=MetaSingleton):
    def __init__(self, stats, x_columns, y_columns) -> None:
        self.stats = stats
        self.x_columns = x_columns
        self.y_columns = y_columns

    @staticmethod
    def create(kind):
        if kind.lower() == "toy":
            return ToySensorDescr()
        elif kind.lower() == "wind_bench":
            return WBSensorDescr()

        raise Exception(f"unable to find descriptions of the sensors {kind}")


class WBSensorDescr(SensorDescr):
    stats = {
        "Tower top fore-aft acceleration ay [m/s2]": {
            "mean": -3.732385818133257e-07,
            "std": 0.1175451058459768,
            "min": -0.8967088086,
            "25%": -0.07215977105999999,
            "50%": 7.251252000000028e-05,
            "75%": 0.07217759763,
            "max": 0.8616838365799999,
        },
        "Tower top side-side acceleration ax [m/s2]": {
            "mean": -4.3043763994239984e-05,
            "std": 0.17042526932798652,
            "min": -1.7410713982200001,
            "25%": -0.08029902833,
            "50%": -5.848058999999989e-05,
            "75%": 0.08026554234000001,
            "max": 1.6791001381199997,
        },
        "Tower mid fore-aft acceleration ay [m/s2]": {
            "mean": 7.059205544883323e-06,
            "std": 0.3120427307662817,
            "min": -3.1043631840000003,
            "25%": -0.1542743856,
            "50%": 0.0,
            "75%": 0.1542152164,
            "max": 2.8346083776,
        },
        "Tower mid side-side acceleration ax [m/s2]": {
            "mean": -1.010584451681167e-05,
            "std": 0.20297381501506764,
            "min": -1.73153678728,
            "25%": -0.0980160628,
            "50%": -6.521140000000439e-06,
            "75%": 0.09805821861,
            "max": 1.7674581912,
        },
        "Tower top rotation x [deg]": {
            "mean": 0.10906119975000214,
            "std": 0.22879160967145254,
            "min": -0.5287499999999454,
            "25%": -0.05624999999997726,
            "50%": 0.114750000000015,
            "75%": 0.2756250000000193,
            "max": 0.757125000000002,
        },
        "Tower top rotation y [deg]": {
            "mean": -0.13664792117511612,
            "std": 0.06582604013556786,
            "min": -0.48334404635999995,
            "25%": -0.186245273584,
            "50%": -0.14719013135999998,
            "75%": -0.08372635442349999,
            "max": 0.02261905472,
        },
        "Tower mid rotation x [deg]": {
            "mean": 0.20820803906250568,
            "std": 0.12007967185727536,
            "min": -0.12374999999997272,
            "25%": 0.12375000000000114,
            "50%": 0.20812499999999545,
            "75%": 0.29812499999999886,
            "max": 0.5343750000000114,
        },
        "Tower mid rotation y [deg]": {
            "mean": -0.03926080885200301,
            "std": 0.02261261187435961,
            "min": -0.21044894401800002,
            "25%": -0.053850658272500004,
            "50%": -0.038773657971,
            "75%": -0.021825153801499998,
            "max": 0.05379317636400001,
        },
        "Mudline moment Mx[kNm]": {
            "mean": 61977.27387588217,
            "std": 29619.62022923741,
            "min": -27907.36832,
            "25%": 40270.44517200001,
            "50%": 61874.379203000004,
            "75%": 83967.554778,
            "max": 152381.0179,
        },
        "Mudline moment My[kNm]": {
            "mean": 7097.506250106164,
            "std": 5128.729624126434,
            "min": -26849.612592,
            "25%": 3270.19544245,
            "50%": 6418.2234465,
            "75%": 10052.65925425,
            "max": 53207.646748,
        },
        "Mudline moment Mz[kNm]": {
            "mean": -711.7177369156096,
            "std": 6796.74697762426,
            "min": -49612.090249999994,
            "25%": -4573.1283393,
            "50%": -535.7095240000001,
            "75%": 3390.6702348500007,
            "max": 43376.417460000004,
        },
        "Waterline moment Mx[kNm]": {
            "mean": 136798.9978638337,
            "std": 68450.42487856085,
            "min": -54995.24456400001,
            "25%": 87216.35613,
            "50%": 136273.3443,
            "75%": 187669.13632800002,
            "max": 330240.99749999994,
        },
        "Waterline moment My[kNm]": {
            "mean": 18307.588403021604,
            "std": 11959.074576665971,
            "min": -49589.05464,
            "25%": 9239.241472999998,
            "50%": 17186.058381000003,
            "75%": 25477.165405999996,
            "max": 119527.84760000001,
        },
        "Waterline moment Mz[kNm]": {
            "mean": -711.7178628248745,
            "std": 6789.644633459557,
            "min": -49547.412064,
            "max": 43301.68051,
        },
    }

    x_columns = [
        "Tower top fore-aft acceleration ay [m/s2]",
        "Tower top side-side acceleration ax [m/s2]",
        "Tower mid fore-aft acceleration ay [m/s2]",
        "Tower mid side-side acceleration ax [m/s2]",
        "Tower top rotation x [deg]",
        "Tower top rotation y [deg]",
        "Tower mid rotation x [deg]",
        "Tower mid rotation y [deg]",
    ]

    y_columns = [
        "Mudline moment Mx[kNm]",
        "Mudline moment My[kNm]",
        "Mudline moment Mz[kNm]",
        "Waterline moment Mx[kNm]",
        "Waterline moment My[kNm]",
        "Waterline moment Mz[kNm]",
    ]

    def __init__(self) -> None:
        super().__init__(
            WBSensorDescr.stats, WBSensorDescr.x_columns, WBSensorDescr.y_columns
        )


class ToySensorDescr(SensorDescr):
    stats = {
        "dof1": {
            "count": 200000.0,
            "mean": 0.6006578321228352,
            "std": 0.4903070154772277,
            "min": -1.6284590340678073,
            "max": 2.628983304050155,
        },
        "dof2": {
            "count": 200000.0,
            "mean": 1.2011745475567646,
            "std": 0.9409551156030835,
            "min": -3.1016032472058237,
            "max": 5.085093939106667,
        },
        "dof3": {
            "count": 200000.0,
            "mean": 1.7516261128345056,
            "std": 1.3183873222793192,
            "min": -4.284208414561359,
            "max": 7.229292830791719,
        },
        "dof4": {
            "count": 200000.0,
            "mean": 2.1519409695808216,
            "std": 1.5943376152804318,
            "min": -5.114358243524277,
            "max": 8.829671276582333,
        },
        "dof5": {
            "count": 200000.0,
            "mean": 2.351882181554723,
            "std": 1.7410202485624346,
            "min": -5.578888241038999,
            "max": 9.69378164120783,
        },
    }

    x_columns = ["dof1", "dof2", "dof4", "dof5"]
    y_columns = ["dof3"]

    def __init__(self) -> None:
        super().__init__(
            ToySensorDescr.stats, ToySensorDescr.x_columns, ToySensorDescr.y_columns
        )
