
"""
This class imputes the median value for each missing indicator
This is just use as benchmark comparison
@Author MeelsL
"""
from copy import deepcopy

import numpy as np

from imputation.models.spline_interpolation import BaseInterpolation


class MedianImputer(BaseInterpolation):

    def __init__(self):
        super().__init__()

    def impute(self, X_test_predict):
        trans_X = X_test_predict.transpose((0, 2, 1))
        mask = np.isnan(trans_X)

        trans_X = deepcopy(trans_X)

        n_samples, n_steps, n_features = mask.shape
        idx = np.where(~mask, np.arange(n_features), 0)

        combiner = []
        #for each sequence
        for x, i in zip(trans_X, idx):
            impute = []
            #for each feature
            for k in range(len(x)):
                # feature_impute = [x[k, j] if i[k, j] != 0 else np.nanmedian(x[k]) for j in range(len(x[k]))]
                feature_impute = x[k]
                idx = np.isnan(x[k])
                x[k][np.all(np.isnan(x[k]))] = 0
                feature_impute[idx] = np.nanmedian(x[k])
                impute.append(feature_impute)
            combiner.append(impute)

        X_imputed = np.asarray(combiner)
        X_imputed = X_imputed.transpose(0, 2, 1)



        return X_imputed

    def __str__(self):
        return "median"