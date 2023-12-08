
import os
import sys
sys.path.append(os.getcwd())
from wb.dataset import FileWBDataset

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input parquet file")
    args = parser.parse_args()
    

    dataset = FileWBDataset(args.input, train_flag=True)

    print(f"train dataset (size) {len(dataset)}")
    print(dataset.partition_keys)

    test_dataset = FileWBDataset(args.input, train_flag=False)

    print(f"test dataset (size) {len(test_dataset)}")
    print(test_dataset.partition_keys)


