import pyarrow.parquet as pq
import json

from wb.dataset.s3 import S3
from wb.dataset import WBDataset


def extract_stats(columns):
    with S3() as s3:
        fs = s3.filesystem
        table = pq.read_table("mpu/wind_bench.parquet", filesystem=fs, columns=columns)
        return table.to_pandas().describe().to_dict()


if __name__ == "__main__":
    stats = extract_stats(WBDataset.x_columns + WBDataset.y_columns)

    with open("stats.json", "w") as f:
        json.dump(stats, f)
