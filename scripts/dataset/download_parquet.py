
import os
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

from wb.dataset.utils.time_utils import  Timer
from wb.dataset.s3 import S3

from args import Args


def download_s3(root_path, parquet_file, n_items):

    with S3() as s3:

        # Computes the number of files in the id partition.
        count = 0
        for obj in s3.bucket.objects.filter(Prefix=parquet_file):
            count += 1

        total = np.min([n_items, count])

        wb_path = os.path.join(root_path, parquet_file)
        os.makedirs(wb_path, exist_ok=True)

        # Download the total files.
        count = 0
        for obj in tqdm(s3.bucket.objects.filter(Prefix=parquet_file), total=total):
            tokens = obj.key.split('/')

            file_directory = os.path.join(wb_path, tokens[1])
            os.makedirs(file_directory, exist_ok=True)
            filename = os.path.join(root_path, obj.key)
            s3.bucket.download_file(obj.key, filename)
            if count == total:
                break
            count += 1


def download_split(root_path, parquet_file, n_items, split):
     
    wb_path = os.path.join(root_path, parquet_file)
    os.makedirs(wb_path, exist_ok=True)

    with tqdm(total=n_items) as pbar:
        with S3() as s3:
            count = 0
            for obj in s3.bucket.objects.filter(Prefix=parquet_file):
                bucket_uri = f"mpu/{obj.key}"

                table = pq.read_table(bucket_uri, filesystem=s3.filesystem)

                n_samples_by_table = table.shape[0] // split
                for i in range(n_samples_by_table):
                    tokens = obj.key.split('/')

                    file_directory = os.path.join(wb_path, f"{tokens[1]}.{i}")
                    os.makedirs(file_directory, exist_ok=True)
                    sub_table = table.slice(i*split, split)
                    pq.write_to_dataset(sub_table, root_path=file_directory)
                    
                    count += 1
                    pbar.update(1)
                    if count >= n_items:
                        return
                    
    print(f"error: you asked {n_items} samples but only {count} are available !")

if __name__ == "__main__":

    args = Args(jsonname = os.path.join(os.path.dirname(__file__), "args.json"))


    download_split(
        root_path=args.data_dir, 
        parquet_file=args.s3_file, 
        n_items=args.n_samples,
        split = args.sequence_length)
        




     