from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from torch.utils.data import DataLoader

from datasets.advanced_loader import Dataset_Pandas
from forecast_impute.models.TransformersBase import TransformersBase
from forecast_impute.models.transformer_module import AdvancedTransformer


class TransformerForecasterTrainer(TransformersBase):

    """
    This class trains the transformer of the AdvancedTransformer class
    This is specifically for forecasting
    It contains training and prediction loops for just forecasting
    @Author MeelsL
    """

    def __init__(self, input_chunk_length: int, decoder_length: int, output_chunk_length: int, resample_rate:str, num_features=1,
                 batch_size: int = 32, n_epochs=100, d_model=64, dim_feedforward=512, num_layers=1, dropout=0.1,
                 imputation=False, advanced_impute=False, diag_mask=False):


        super().__init__(input_chunk_length, decoder_length, output_chunk_length, resample_rate, num_features,
                 batch_size, n_epochs, d_model, dim_feedforward, num_layers, dropout, imputation, advanced_impute)

        task = "long_term_forecast"

        self.model = AdvancedTransformer(num_features = num_features, input_chunk_length=input_chunk_length,
                                         decoder_length=decoder_length, output_chunk_length=output_chunk_length, d_model= d_model, dim_feedforward= dim_feedforward,
                                         num_layers = num_layers, dropout= dropout,
                                         task_name=task, device=self.device, advanced_impute = advanced_impute, resample_rate=resample_rate, diag_mask=diag_mask)

        self.model = self.model.to(self.device)



    def fit(self, train_data : TimeSeries):
        """
        Fit the model only for forecasting
        :param train_data: training data
        :return: trained model
        """
        if isinstance(train_data, pd.DataFrame):
            train_data = Dataset_Pandas(train_data, input_chunk_length=self.input_chunk_length,
                                        decoder_length=self.decoder_length, output_chunk_length=self.output_chunk_length,
                                        freq=self.resample_rate, imputation = self.imputation, missing_rate=0)

        if isinstance(train_data, TimeSeries):
            time_index = train_data.time_index
            train_data = np.squeeze(train_data.all_values())

            train_data = Dataset_Pandas(train_data, input_chunk_length=self.input_chunk_length,
                                        decoder_length=self.decoder_length, output_chunk_length=self.output_chunk_length, freq=self.resample_rate,
                                        time_index = time_index, imputation=self.imputation, missing_rate=0)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

        return self.fit_loop(train_data, train_loader)

    def predict(self, n_steps, series, val_series, covariates=None, covariates_val=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict when there is no imputation involved
        :param n_steps: number of steps to predict into future
        :param series: train series
        :param val_series: validation series
        :param covariates: covariates
        :param covariates_val: covariates in validation
        :return: prediction and target series
        """

        n_steps = len(val_series)
        divider = int(n_steps/self.output_chunk_length)
        n_steps = divider*self.output_chunk_length

        if isinstance(series, pd.DataFrame):
            input_data = pd.concat([series.iloc[-self.input_chunk_length:], val_series])

            input_data = Dataset_Pandas(input_data, input_chunk_length=self.input_chunk_length,
                                        decoder_length=self.decoder_length,
                                        output_chunk_length=self.output_chunk_length, freq=self.resample_rate, imputation=self.imputation,
                                        missing_rate=0)
        if isinstance(series, TimeSeries):
            time_index = series.time_index.union(val_series.time_index)
            series = np.squeeze(series.all_values())
            val_series = np.squeeze(val_series.all_values())

            input_data = np.concatenate((series[-self.input_chunk_length:], val_series))
            time_index = pd.DatetimeIndex(pd.Series(time_index).iloc[-len(input_data):])
            input_data = Dataset_Pandas(input_data, input_chunk_length=self.input_chunk_length,
                                        decoder_length=self.decoder_length,
                                        output_chunk_length=self.output_chunk_length, freq=self.resample_rate,
                                        time_index=time_index, imputation=self.imputation,
                                        missing_rate=0)




        input_loader = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return self.predict_loop(n_steps, input_loader)