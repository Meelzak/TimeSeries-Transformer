from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from datasets.advanced_loader import Dataset_from_arrays
from forecast_impute.models.transformer_module import AdvancedTransformer
from imputation.models.transformer.transformer_imputer_trainer import TransformerImputerTrainer


class TransformerForecastImputeTrainer(TransformerImputerTrainer):

    """
    This class trains the transformer of the AdvancedTransformer class
    This is specifically for forecasting and imputation
    It contains training and prediction loops for both imputation and forecasting
    @Author MeelsL
    """

    def __init__(self, input_chunk_length: int, decoder_length: int, output_chunk_length: int, resample_rate:str, num_features=1,
                 batch_size: int = 32, n_epochs=100, d_model=64, dim_feedforward=512, num_layers=1, dropout=0.1,
                 imputation=False, advanced_impute=False, diag_mask=True):


        super().__init__(input_chunk_length, decoder_length, output_chunk_length, resample_rate, num_features,
                 batch_size, n_epochs, d_model, dim_feedforward, num_layers, dropout, imputation, advanced_impute, diag_mask)




    def fit_impute(self, data_to_impute, data_y, time_index:pd.DatetimeIndex, missing_rate:float):
        """
        fit the model for BOTH forecasting and imputation
        :param data_to_impute: data with missing values
        :param data_y: data with true values
        :param time_index: timestep of the datapoints
        :param missing_rate: missing rate to artifically generate missing values
        :return: trained model
        """

        train_data = Dataset_from_arrays(data_x=data_to_impute, data_y=data_y, input_chunk_length=self.input_chunk_length,
                                         decoder_length=self.decoder_length, output_chunk_length=self.output_chunk_length,
                                         freq=self.resample_rate, imputation=self.imputation,
                                         time_index=time_index, missing_rate=missing_rate)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

        # self.model.task_name = "imputation"
        # self.fit_simple(train_data, train_loader)

        self.model.task_name = "impute_and_forecast"
        return self.fit_loop(train_data, train_loader)

    def predict_impute(self, data_to_impute, data_y, time_index:pd.DatetimeIndex)->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict when there are missing values in the data
        :param data_to_impute: data with missing values
        :param data_y:  data with no missing values
        :return: forecast and target values
        """
        n_steps = len(data_y) - self.input_chunk_length

        divider = int(n_steps/self.output_chunk_length)
        n_steps = divider*self.output_chunk_length

        input_data = Dataset_from_arrays(data_x=data_to_impute, data_y=data_y, input_chunk_length=self.input_chunk_length,
                                         decoder_length=self.decoder_length, output_chunk_length=self.output_chunk_length,
                                         freq=self.resample_rate, imputation=self.imputation, time_index=time_index,
                                         missing_rate=0)

        input_loader = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

        return self.predict_loop(n_steps, input_loader)