import h5py
import numpy as np
import pandas as pd
from pypots.data import load_specific_dataset, mcar, masked_fill
from sklearn.preprocessing import StandardScaler


"""
This class loads some benchmark datasets for timeseries imputation

@Author MeelsL
"""
def load_air_quality_dataset(missing_rate:float):
    """
    loads the air quality dataset
    original missing rate: 1.6%
    n_steps required for model = 24
    data is already processed so no normalisation is required anymore
    :param missing_rate to artificially generate missing data
    :return: train, val, test datasets
    """
    with h5py.File("../datasets/Air Quality/datasets.h5", "r") as f:
        X_train = f['train']['X'][:]
        X_val = f['val']['X'][:]
        X_test_complete = f['test']['X'][:]

    X_test_complete, X_test_missing, missing_mask, X_test_indicating_mask = mcar(X_test_complete, missing_rate)
    X_test_missing = masked_fill(X_test_missing, 1 - missing_mask, np.nan)

    n_steps = 24

    X_train, X_val, X_test_complete, X_test_missing, X_test_indicating_mask = X_train[:, :, :33], X_val[:, :, :33], X_test_complete[:, :, :33], X_test_missing[:, :, :33], X_test_indicating_mask[:, :, :33]
    return X_train, X_val, X_test_complete, X_test_missing, X_test_indicating_mask, n_steps


