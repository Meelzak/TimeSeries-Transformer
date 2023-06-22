import math
from typing import Tuple

import numpy as np


def create_missing_chunks(X, chunk_rate, chunk_size, nan=0):
    """
    generates artificial missing data of a certain chunk size
    In the case of sensors it will randomly remove values for a certain period of time
    :param X: original dataset
    :param chunk_rate: probability that a chunk will be removed
    :param chunk_size: the size of the chunk to remove
    :param nan: which value to fill in the missng values
    :return: Original sequence, sequence with missing data, indicators if missing or not
    """

    original_shape = X.shape
    try:
        num_features = original_shape[2]
    except IndexError:
        num_features = original_shape[1]

    X = X.flatten()
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact

    # select random indices for artificial mask
    indices = np.where(~np.isnan(X))[0].tolist()  # get the indices of observed values
    indices = np.random.choice(indices, int((len(indices) * chunk_rate)/chunk_size), replace=False)

    indices_range = np.zeros(shape=(len(indices),chunk_size))

    for i in range(len(indices)):
        if indices[i] + chunk_size*num_features < len(X):
            indices_range[i] = np.arange(start=indices[i], stop=indices[i]+chunk_size*num_features, step=num_features)

    indices_range = indices_range.flatten()
    indices_range = indices_range.astype(np.int)

    # create artificially-missing values by selected indices
    X[indices_range] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)

    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)
    return X_intact, X, missing_mask, indicating_mask


def create_all_sensors_missing_chunks(X, chunk_rate, chunk_size, nan=0):
    """
    Creates missing chunks but such that chunks cover all sensors
    Meaning there is missing data for ALL sensors at a given timestep
    :param X: original dataset
    :param chunk_rate: probability that a chunk will be removed
    :param chunk_size: the size of the chunk to remove
    :param nan: which value to fill in the missng values
    :return: Original sequence, sequence with missing data, indicators if missing or not
    """

    original_shape = X.shape
    try:
        num_features = original_shape[2]
    except IndexError:
        num_features = original_shape[1]


    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact


    indices = np.random.choice(np.arange(len(X_intact[:,0])), int((len(X_intact[:,0]) * chunk_rate)/chunk_size), replace=False)

    indices_range = np.zeros(shape=original_shape)

    for i in range(len(indices)):
        if indices[i] + chunk_size < len(X):
            indices_range[int(indices[i]):int(indices[i])+chunk_size] = np.ones(shape=(chunk_size,num_features))

    # create artificially-missing values by selected indices
    X[indices_range.astype(bool)] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)

    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)
    return X_intact, X, missing_mask, indicating_mask


def compute_nan_ratio(X_missing) -> Tuple[float, float]:
    """
    compute mean and standard deviation of the original nan values
    These values are stored in global variables such that they can be accessed anywhere in the code
    :param X_missing:
    :return: mean, std float
    """

    # make everything 1d
    original_shape = X_missing.shape
    X_missing = np.nan_to_num(X_missing, nan=0)
    X_missing = X_missing.reshape(-1, original_shape[1])
    X_missing = X_missing.flatten('F')

    # count consecutive zeros
    iszero = np.concatenate(([0], np.equal(X_missing, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    distribution = ranges[:, 1] - ranges[:, 0]

    global mean_chunk_size
    mean_chunk_size = np.mean(distribution)
    if math.isnan(mean_chunk_size):
        mean_chunk_size = 24

    global std_chunk_size
    std_chunk_size = np.std(distribution)

    if math.isnan(std_chunk_size):
        std_chunk_size = 12

    print("mean missing: ", mean_chunk_size, "std missing: ", std_chunk_size)

    return mean_chunk_size, std_chunk_size


def create_variable_missing_chunks(X, chunk_rate, nan=0):
    """
    generates artificial missing data of a certain chunk size
    chunk size varies depending on a distribution
    this is based on the original missing rate of the data
    In the case of sensors it will randomly remove values for a certain period of time
    :param X: original dataset
    :param chunk_rate: probability that a chunk will be removed
    :param nan: which value to fill in the missng values
    :return: Original sequence, sequence with missing data, indicators if missing or not
    """

    original_shape = X.shape
    try:
        num_features = original_shape[2]
    except IndexError:
        num_features = original_shape[1]

    X = X.flatten()
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact

    # select random indices for artificial mask
    indices = np.where(~np.isnan(X))[0].tolist()# get the indices of observed values
    indices_range = np.array([])

    #sample till threshold is reached
    while len(indices_range)/len(indices)<chunk_rate:
        sample_num = int(chunk_rate*len(indices)/(mean_chunk_size+1.5*std_chunk_size))

        #randomly sample starting index
        sub_indices = np.random.choice(indices, sample_num, replace=False)

        #draw chunk length of distribution
        chunk_length = np.random.normal(mean_chunk_size, std_chunk_size, sample_num).astype(int)

        for i in range(len(sub_indices)):
            if sub_indices[i] + chunk_length[i] * num_features < len(X) and chunk_length[i]>0:
                indices_range = np.concatenate((indices_range, (np.arange(start=sub_indices[i], stop=sub_indices[i]+chunk_length[i]*num_features, step=num_features))))

        #no duplicates are allowed to mask
        indices_range = np.unique(indices_range)


    indices_range = indices_range.astype(np.int)

    # create artificially-missing values by selected indices
    X[indices_range] = np.nan  # mask values selected by indices
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)

    # reshape into time-series data
    X_intact = X_intact.reshape(original_shape)
    X = X.reshape(original_shape)
    missing_mask = missing_mask.reshape(original_shape)
    indicating_mask = indicating_mask.reshape(original_shape)

    return X_intact, X, missing_mask, indicating_mask