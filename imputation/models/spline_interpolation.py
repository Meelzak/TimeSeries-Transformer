import numpy as np
import pandas as pd

"""
Contains simple interpolation methods
All methods use the same interface such that the models can be used in a standardised way
@Author MeelsL
"""

class BaseInterpolation():
    """
    Interface method used as standarised method
    """

    def __init(self):
        pass

    def fit(self, X_train, X_val=None):
        return self

    def __str__(self):
        return "Base interpolation"


class SplineInterpolation(BaseInterpolation):

    def __init__(self, method="spline"):
        """
        This uses pandas interpolation method
        Inbuilt it uses the scipy interpolation algorithm which also includes spline, or polynomial of higher order polynomials
        :param method: str (spline, polynomial or linear)
        """
        super().__init__()

        self.method = method


    def impute(self, X_test_predict):
        original_shape = X_test_predict.shape
        num_features = X_test_predict.shape[2]

        X_test_predict = X_test_predict.reshape(-1, num_features)

        df = pd.DataFrame(data=X_test_predict, index=np.arange(len(X_test_predict)),
                          columns=np.arange(len(X_test_predict[0])))

        if self.method == "linear":
            df = df.interpolate(method=self.method)
        else:
            df = df.interpolate(method=self.method, order=3)

        df = df.fillna(df.median().median())
        return df.values.reshape(original_shape)

    def __str__(self):
        return self.method