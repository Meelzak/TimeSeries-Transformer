from darts import TimeSeries
from pmdarima import auto_arima as PmdARIMA
from pmdarima import ARIMA
class AutoARIMA():
    """
    AutoARIMA model using pmdARIMA package
    Method uses a standardised version for easier benchmarking
    """

    def __init__(self, start_p, end_p, start_q, end_q, start_d, end_d):
        self.start_p = start_p
        self.end_p = end_p
        self.start_q = start_q
        self.end_q = end_q
        self.start_d = start_d
        self.end_d = end_d


    def fit(self, series:TimeSeries):
        # model = PmdARIMA(y = series.values(), start_p=self.start_p, end_p = self.end_p, start_q = self.start_q, end_q= self.end_q, seasonal=True, stepwise=False, n_fits=500, trace=True, maxiter=500,
        #                  max_order=200, m=24)

        model = ARIMA(y= series.values(), order=(2,0,1), seasonal_order=(2,0,1,24*1))
        model.fit(series.values())

        self.model = model

    def predict(self, prediction_length:int):
        return TimeSeries.from_values(self.model.predict(prediction_length))

