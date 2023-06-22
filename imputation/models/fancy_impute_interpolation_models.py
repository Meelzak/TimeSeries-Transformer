import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

from fancyimpute import IterativeSVD, MatrixFactorization, KNN

from imputation.models.spline_interpolation import BaseInterpolation


"""
Interpolation methods based on the fancyimpute library
@Author MeelsL
"""
class KNN_imputation(BaseInterpolation):

    def __init__(self, k:int):
        super().__init__()
        self.k = k


    def impute(self, X_test_predict):
        original_shape = X_test_predict.shape
        num_features = X_test_predict.shape[2]

        X_test_predict = X_test_predict.reshape(-1, num_features)

        index = np.arange(len(X_test_predict))
        df = pd.DataFrame(data=X_test_predict, index=index,
                          columns=np.arange(len(X_test_predict[0])))

        df_impute = KNN(k=self.k).fit_transform(df)

        return df_impute.reshape(original_shape)

    def __str__(self):
        return "knn"


class SingularValueDecomposition(BaseInterpolation):

    def __init__(self):
        super().__init__()


    def impute(self, X_test_predict):
        original_shape = X_test_predict.shape
        num_features = X_test_predict.shape[2]

        X_test_predict = X_test_predict.reshape(-1, num_features)

        index = np.arange(len(X_test_predict))
        df = pd.DataFrame(data=X_test_predict, index=index,
                          columns=np.arange(len(X_test_predict[0])))

        df_impute = IterativeSVD().fit_transform(df)

        return df_impute.reshape(original_shape)

    def __str__(self):
        return "SVD"

class MatrixFactorizationImputation(BaseInterpolation):

    def __init__(self):
        super().__init__()

    def impute(self, X_test_predict):
        original_shape = X_test_predict.shape
        num_features = X_test_predict.shape[2]

        X_test_predict = X_test_predict.reshape(-1, num_features)

        index = np.arange(len(X_test_predict))
        df = pd.DataFrame(data=X_test_predict, index=index,
                          columns=np.arange(len(X_test_predict[0])))

        df_impute = MatrixFactorization().fit_transform(df)

        return df_impute.reshape(original_shape)

    def __str__(self):
        return "MF"