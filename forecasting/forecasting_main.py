from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TransformerModel, NBEATSModel, NHiTSModel, TFTModel, VARIMA, XGBModel, KalmanForecaster, \
    RegressionModel, StatsForecastAutoARIMA, ExponentialSmoothing, Prophet, NaiveSeasonal, LightGBMModel, \
    LinearRegressionModel, ARIMA, RNNModel, BlockRNNModel, Theta, FFT


import torch
from darts.utils.statistics import check_seasonality
from darts.utils.utils import SeasonalityMode, ModelMode
from pypots.utils.metrics import cal_mre
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.sensor_data_loading import _multiple_sensor_temperature, no_nan_sensors, _load_sensor_csv, \
    forecast_impute_sensors

from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

from forecast_impute.models.AutoARIMA import AutoARIMA
from forecasting.models.transformer_forecaster_trainer import TransformerForecasterTrainer



"""
This file is the main class for testing all forecasting methods
@Author MeelsL
"""

def create_subset(sensor_data:pd.DataFrame, sample_rate='d')->(TimeSeries, TimeSeries, StandardScaler):
    """
    Creates a training and validation dataset
    Missing data is filled with median value
    Data is set to a fixed frequency (sample rate) and missing values are filled by mean value
    Data is standardised using StandardScalar
    :param sensor_data: dataframe data
    :param sample_rate: rate to fix the frequency
    :return: training set, validation set, and scaler used for transformation
    """

    # Resample the input data to daily intervals using the mean of each day
    df = sensor_data.resample(sample_rate).mean(numeric_only=True)


    try:
        median = df.median().median()
    except AttributeError:
        median = df.median()

    df = df.fillna(median)

    scaler = StandardScaler()
    array_version = scaler.fit_transform(df) #df.to_numpy().reshape(-1,1)

    df = pd.DataFrame(data=array_version, index=df.index, columns=df.columns) #[1]
    series = TimeSeries.from_dataframe(df, fillna_value=median)



    series_train, series_val = series.split_before(0.55) #pd.Timestamp('2023-01-01') #2009-01-01
    series_val, _ = series_val.split_before(0.75) #pd.Timestamp('2023-02-18')

    return series_train, series_val, scaler

def check_seasonality_significance(series:TimeSeries):
    """
    checks if the daily and weekly seasonality is statistically significant
    :param series: series to check
    :return: prints if series seasonality is statistically significant
    """

    #check can only be performed on a univariate series
    series = series.univariate_component(0)

    is_daily_seasonal, daily_period = check_seasonality(series, m=24, max_lag=400, alpha=0.05)
    is_weekly_seasonal, weekly_period = check_seasonality(series, m=168, max_lag=400, alpha=0.05)

    print(f'Daily seasonality: {is_daily_seasonal} - period = {daily_period}')
    print(f'Weekly seasonality: {is_weekly_seasonal} - period = {weekly_period}')


def train_evaluate_multiple_univariate_models(models:list, train_series:TimeSeries, val_series:TimeSeries):
    """
    trains multiple univariate dart models
    :param models: list of univariate models
    :param train_series: series to train the model on
    :param val_series: validation series to measure performance
    :return:
    """

    for model in models:
        predictions = None
        for i in tqdm(train_series.columns):
            series = train_series.univariate_component(i)
            model.fit(series)
            if predictions is None:
                predictions = model.predict(len(val_series))
            else:
                predictions = predictions.concatenate(model.predict(len(val_series)), ignore_time_axis=True, axis=1)



        print(model)
        r2 = r2_score(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()))
        mse = mean_squared_error(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()))
        rmse = mean_squared_error(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()), squared=False)
        mae = mean_absolute_error(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()))
        mape = mean_absolute_percentage_error(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()))

        print(f"R2_score: {r2}, mse: {mse}, rmse: {rmse}, mae: {mae}, mape: {mape}")

        plot_forecast(np.squeeze(val_series.all_values()), np.squeeze(predictions.all_values()), val_series.time_index)


def select_univariate_models():
    """
    :return:a list of univariate darts models to test
    """

    smooth = ExponentialSmoothing(seasonal_periods=24*7, damped=True, trend=ModelMode.ADDITIVE)
    prophet = Prophet()

    fft = FFT(nr_freqs_to_keep=20, trend=None)

    arima = AutoARIMA(start_p=5, end_p=14, start_q=5, end_q=14, start_d=0, end_d=2)
    univariate_models = [smooth, prophet, fft, arima]



    return univariate_models

def select_multivariate_models(resample_rate:str):
    """
    :param resample_rate: specifiy if series has daily or hourly timestamps
    :return: a list of multivariate darts models to check
    """
    forecast_window = 1
    if resample_rate == "h":
        forecast_window = 24


    transformer = TransformerModel(input_chunk_length=14*forecast_window, output_chunk_length=1*forecast_window,  # layer_widths=32,
                             batch_size=32, n_epochs=10,
                             d_model=64, #512
                             # # nhead=4, #8
                             # num_encoder_layers= 4, #6
                             # num_decoder_layers= 4, #6
                             # dim_feedforward= 128, #2048
                             dropout=0.1,

                             pl_trainer_kwargs={
                                 "accelerator": "gpu",
                                 "devices": [0]
                             },
                             )

    nbeats = NBEATSModel(input_chunk_length=7*24, output_chunk_length=1*24, #layer_widths=32,
                        batch_size = 32, n_epochs=10,
                        pl_trainer_kwargs={
                            "accelerator": "gpu",
                            "devices": [0],
                            # "callbacks": [my_stopper]
                        },
                        )

    nhits = NHiTSModel(input_chunk_length=7*24, output_chunk_length=1*24, #layer_widths=32,
                        batch_size = 32, n_epochs=10,
                        pl_trainer_kwargs={
                            "accelerator": "gpu",
                            "devices": [0],
                            # "callbacks": [my_stopper]
                        },
                        )

    rnn = BlockRNNModel(
        model="LSTM",
        input_chunk_length=7*24,
        output_chunk_length=1*24,
        hidden_dim=64,
        n_rnn_layers=1,
        n_epochs=10,
        batch_size=32,
        loss_fn=torch.nn.MSELoss(),
        pl_trainer_kwargs={
                            "accelerator": "gpu",
                            "devices": [0]
                        },
        model_name="lstm_model",
    )

    temporal_fusion = TFTModel(input_chunk_length=14*forecast_window, output_chunk_length=1*forecast_window,  # layer_widths=32,
                             batch_size=32, n_epochs=10,
                             hidden_size=16,
                             dropout=0.1,
                               add_encoders={
                                   'cyclic': {'future': ['day', 'week']},
                                   'datetime_attribute': {'future': ['hour', 'dayofweek']},
                                   'position': {'past': ['relative'], 'future': ['relative']}
                               },

                             pl_trainer_kwargs={
                                 "accelerator": "gpu",
                                 "devices": [0]
                             },)

    naive = NaiveSeasonal(K=7*24)
    xgb = XGBModel(lags=[-24*7, -24])
    linear = LinearRegressionModel(lags = [-24*7, -24])

    multivariate_models = [naive, xgb, linear, rnn, nbeats, nhits, transformer, temporal_fusion]
    return multivariate_models


def train_evaluate_multivariate_models(models:list, train_series:TimeSeries, val_series:TimeSeries):
    """
    trains multiple multivariate darts models
    :param models: list of models
    :param train_series: series to train models
    :param val_series: series to test models
    :return: models performance and plots prediction
    """

    for model in models:
        model.fit(train_series)
        prediction = model.predict(len(val_series))


        val_series_array = np.squeeze(val_series.all_values())
        prediction_array = np.squeeze(prediction.all_values())

        print(model)
        r2 = r2_score(val_series_array, prediction_array)
        mse = mean_squared_error(val_series_array, prediction_array)
        rmse = mean_squared_error(val_series_array, prediction_array,
                                  squared=False)
        mae = mean_absolute_error(val_series_array, prediction_array)
        mape = mean_absolute_percentage_error(val_series_array, prediction_array)
        mre = cal_mre(prediction_array, val_series_array)

        print(f"R2_score: {r2}, mse: {mse}, rmse: {rmse}, mae: {mae}, mape: {mape}, mre: {mre}")

        plot_forecast(val_series_array, prediction_array, val_series.time_index)

def train_proposed_architecture(train_series:TimeSeries, resample_rate = 'd'):
    """
    trains my custom transformer model
    :param train_series: series to train model on
    :param resample_rate: specify if data has daily or hourly timestamps
    :return: trained model
    """

    forecast_window = 1
    if resample_rate == "h":
        forecast_window = 24

    num_features = train_series.all_values().shape[1]




    model = TransformerForecasterTrainer(input_chunk_length=7*forecast_window, decoder_length=4*forecast_window, output_chunk_length=1*forecast_window, num_features=num_features,
                                  n_epochs=10,
                                  batch_size=32,
                                  d_model= 64,
                                  dim_feedforward=512,
                                  num_layers=1,
                                imputation=False,
                               resample_rate=resample_rate)


    model.fit(train_series)

    return model



def evaluate_proposed_architecture(model, train_series:TimeSeries, val_series:TimeSeries, time_index:pd.DatetimeIndex=None):
    """
    evaluates proposed architecture
    :param model: transformer model
    :param train_series: train dataseries
    :param val_series: validation series
    :param time_index: time index
    :return: mse, rmse, mae, mape score and plot
    """

    prediction, truth, _ = model.predict(len(val_series), series=train_series, val_series=val_series, covariates=0, covariates_val=0)
    train_series_array = np.squeeze(train_series.all_values())
    val_series_array = truth

    time_index = time_index[:(len(train_series_array) + len(val_series_array))]



    r2 = r2_score(val_series_array, prediction)
    mse = mean_squared_error(val_series_array, prediction)
    rmse = mean_squared_error(val_series_array, prediction, squared=False)
    mae = mean_absolute_error(val_series_array, prediction)
    mape = mean_absolute_percentage_error(val_series_array, prediction)

    print(f"R2_score: {r2}, mse: {mse}, rmse: {rmse}, mae: {mae}, mape: {mape}")

    plot_forecast(val_series_array, prediction, time_index[-len(prediction):])
    # plot_forecast(np.concatenate((train_series_scaled, val_series_scaled)), prediction, time_index)

def plot_forecast(series:np.ndarray, prediction:np.ndarray, time_index:pd.DatetimeIndex=None):
    """
    Plot the forecast of a model for a single sensor.

    :param series: A complete time series with correct values.
    :param prediction: Prediction value computed by a model.
    :param time_index: An array with time indices corresponding to the `series` and `prediction` arrays.

    :return: plot
    """

    if time_index is None:
        time_index = np.arange(len(prediction))

    fig = px.line(title="sensor data real vs predicted values", color_discrete_sequence=["blue"])
    try:
        fig.add_scatter(x=time_index, y=series[:,0], name="actual")
        fig.add_scatter(x=time_index[-(len(prediction)):], y=prediction[:,0], name="prediction")
    except IndexError:
        fig.add_scatter(x=time_index, y=series, name="actual")
        fig.add_scatter(x=time_index[-(len(prediction)):], y=prediction, name="prediction")
    fig.show()


if __name__ == '__main__':
    resample_rate = 'h'

    "select dataset"
    sensor_data = forecast_impute_sensors(resample_rate)

    train_series, val_series, scaler = create_subset(sensor_data, sample_rate=resample_rate)

    check_seasonality_significance(train_series)

    "train univariate models"
    univariate_models = select_univariate_models()
    train_evaluate_multiple_univariate_models(univariate_models, train_series, val_series)

    "train multivariate models"
    multivariate_models = select_multivariate_models(resample_rate)
    train_evaluate_multivariate_models(multivariate_models, train_series, val_series)


    "train proposed architecutre"
    model = train_proposed_architecture(train_series, resample_rate)
    index = train_series.time_index.union(val_series.time_index)
    evaluate_proposed_architecture(model, train_series, val_series, index)
