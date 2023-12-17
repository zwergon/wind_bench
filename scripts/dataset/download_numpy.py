import os
import pyarrow.parquet as pq
import numpy as np
from sklearn.model_selection import train_test_split

from wb.dataset.s3 import S3
from wb.dataset import WBDataset
from wb.utils.config import Config
from tqdm import tqdm


def dummy_array():
    data = np.zeros(shape=(3, 10, 2))
    for i in range(data.shape[0]):
        data[i, :, 0] = np.random.normal(1, 0.4, 10)
        data[i, :, 1] = np.random.normal(0, 2, 10)

    return data


def download_split(parquet_file, n_items, split):
    main_columns = WBDataset.x_columns + WBDataset.y_columns

    array = np.zeros(shape=(n_items, split, len(main_columns)), dtype=np.float32)

    with S3() as s3:
        count = 0

        with tqdm(total=n_items) as pbar:
            for obj in s3.bucket.objects.filter(Prefix=parquet_file):
                bucket_uri = f"mpu/{obj.key}"

                table = pq.read_table(
                    bucket_uri, columns=main_columns, filesystem=s3.filesystem
                )

                n_samples_by_table = table.shape[0] // split
                for i in range(n_samples_by_table):
                    sub_table = table.slice(i * split, split).to_pandas().to_numpy()
                    array[count, :, :] = sub_table

                    count += 1
                    pbar.update(1)
                    if count >= n_items:
                        return array

    print(f"error: you asked {n_items} samples but only {count} are available !")

    return None


if __name__ == "__main__":
    config = Config(jsonname=os.path.join(os.path.dirname(__file__), "config.json"))

    data = download_split(config.s3_file, config.n_samples, config.sequence_length)

    if config.normalization:
        min = np.min(data, axis=(0, 1))
        max = np.max(data, axis=(0, 1))
        for i in range(data.shape[0]):
            data[i, :, :] = (data[i, :, :] - min) / (max - min)

    idx_y = len(WBDataset.y_columns)
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data[:, :, :-idx_y],
        data[:, :, -idx_y:],
        test_size=config.ratio_train_test,
        random_state=42,
    )

    X_train = np.swapaxes(X_train, 1, 2)
    y_train = np.swapaxes(y_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    y_test = np.swapaxes(y_test, 1, 2)

    print(f"X_train : {X_train.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")

    # Save train and test sets to separate files
    # Enregistrer les données d'entraînement

    out_dir = config.data_dir
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train_data.npy"), X_train)
    np.save(os.path.join(out_dir, "train_labels.npy"), y_train)

    # Enregistrer les données de test
    np.save(os.path.join(out_dir, "test_data.npy"), X_test)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)
