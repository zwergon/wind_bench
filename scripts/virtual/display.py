
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from wb.dataset import dataset

from wb.utils.config import Config

if __name__ == "__main__":
    config = Config(jsonname = os.path.join(os.path.dirname(__file__), "config.json"))

    dataset, _ = dataset(config)
        
    print(f"train dataset (size) {len(dataset)}")

    X_idx, y_idx = dataset[config.indices[0]]

    print(f"X_test : {X_idx.shape}")
    print(f"y_test : {y_idx.shape}")

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle( f"X (solid) versus y (dash) for idx:{config.indices[0]}")

    print(X_idx.shape, y_idx.shape)
    for i in range(X_idx.shape[0]):
        ax1.plot(X_idx[i, :])
    
    for i in range(y_idx.shape[0]):
        ax2.plot(y_idx[i, :], "--")
   
    plt.show()

    