[![License](https://img.shields.io/badge/license-MIT-white)](
    https://stringfixer.com/fr/MIT_license)
[![Flake8](
    https://github.com/zwergon/wind_bench/actions/workflows/python-app.yml/badge.svg)](
        https://github.com/zwergon/wind_bench/actions/workflows/python-app.yml)

![mlflow](https://img.shields.io/badge/mlflow-%23d9ead3.svg?style=for-the-badge&logo=numpy&logoColor=blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

# Wind Bench

This repository contains a python module for testing different neural network architectures (CNN1D, LSTM, RNN, Resnet1D, Unet1D), which aim to estimate the temporal output of virtual force sensors from temporal signals from accelerometric sensors.

| <img src="./assets/images/virtual_sensing_aim.png" >



# Quick start

### Installation for developers

Install the module and dependencies in a virtual environment with Python 3.7-3.10.

```bash
pip install -e .
pip install -r requirements.txt
```

### Train on the test dataset

A small dataset is available by default in this repository. It contains 100
timeseries of size 128 samples. 

You can simply run a train on this dataset from root folder:

```bash
python scripts/virtual/train.py tests/data/100_128/wind_bench.parquet 
```

---

Date: 2023-12-17

Author: [github@zwergon](https://github.com/zwergon)

Copyright © 2022 Jean-François Lecomte

MIT License ([see here](LICENSE.md))

---