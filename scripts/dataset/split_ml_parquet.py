import os
import pyarrow.parquet as pq

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from tqdm import tqdm


def _search_parquets(rootdir):
    file_list = []
    for root, directories, file in os.walk(rootdir):
        for file in file:
            if(file.endswith(".parquet")):
                file_list.append(os.path.join(root, file))
    return file_list



def download_split(in_parquet_file, out_parquet_file, n_items, split):

    os.makedirs(out_parquet_file, exist_ok=True)

    count = 0

    
    for file in _search_parquets(in_parquet_file):
        
            table = pq.read_table(file) 
        
            n_samples_by_table = table.shape[0] // split
            for i in range(n_samples_by_table):
                tokens = file.split('/')

                file_directory = os.path.join(out_parquet_file, f"{tokens[-2]}.{i}")
                print(file_directory)
                os.makedirs(file_directory, exist_ok=True)
                sub_table = table.slice(i*split, split)
                pq.write_to_dataset(sub_table, root_path=file_directory)
                
                count += 1
                if count >= n_items:
                    return
                    
    print(f"error: you asked {n_items} samples but only {count} are available !")

if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="input parquet file")
    parser.add_argument("-o", "--output", help="output_parquet_file")
    parser.add_argument("-n", "--nitems", help='n samples in dataset', type=int, default=1000)
    parser.add_argument("-s", "--size", type=int, help="size of one sample in the dataset", default=1028)
    args = parser.parse_args()

    
    download_split(args.input, args.output, n_items=args.nitems, split=args.size)
