from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import TransformerModel
from pypots.data import mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae, cal_mse, cal_rmse, cal_mre
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

from datasets.impute_dataset import create_missing_chunks, create_all_sensors_missing_chunks, compute_nan_ratio, \
    create_variable_missing_chunks
from datasets.sensor_data_loading import forecast_impute_sensors
from forecast_impute.models.transformer_forecast_impute_trainer import TransformerForecastImputeTrainer
import plotly.express as px

from forecasting.forecasting_main import select_univariate_models, select_multivariate_models

"""
Main class for testing forecasting methods with missing input data
"""
def train_test_split(sensor_data:pd.DataFrame, missing_rate : float)->Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    splits the data in train, test set and artificially removes data
    :param sensor_data: training data
    :param missing_rate: fraction to generate missing values
    :return: train_intact, test_intact, train_missing, test_missing
    """

    scaler = StandardScaler()
    X = scaler.fit_transform(sensor_data.to_numpy())

    mean, std = compute_nan_ratio(X)

    #fill original nan values
    # X = np.nan_to_num(X, nan=float(np.nanmedian(X)))

    X_intact, X_missing, missing_mask, indicating_mask = create_variable_missing_chunks(X, missing_rate)

    artficial_rate = np.count_nonzero(X_missing)
    real_rate = np.count_nonzero(X_intact)
    print("artificial missing length: ", artficial_rate)
    print("original missing length: ", real_rate)
    X_missing = masked_fill(X_missing, 1 - missing_mask, np.nan)

    X_intact_train, X_intact_test = X_intact[:-int(len(X_intact)*0.3)], X_intact[-int(len(X_intact)*0.3):-int(len(X_intact)*0.1)]
    X_missing_train, X_missing_test = X_missing[:-int(len(X_missing) * 0.3)], X_missing[-int(len(X_intact)*0.3):-int(len(X_intact)*0.1)]


    index = sensor_data.index[:len(X_missing_train)+len(X_intact_test)]

    return X_intact_train, X_intact_test, X_missing_train, X_missing_test, index

def train_saits_imputer( X_intact_train, X_intact_test, X_missing_train, X_missing_test, index:pd.DatetimeIndex):
    """
    trains an imputer for the forecasting methods that cannot handle missing data by itself
    quality of the imputation is major factor in forecasting performance
    :param X_intact_train: intact training sequence
    :param X_intact_test: intact test sequence
    :param X_missing_train: training sequence which need to be imputed
    :param X_missing_test: test sequence which need to be imputed
    :return: imputed representation of X_train and X_test
    """

    num_features = X_intact_train.shape[1]

    #train imputer
    #make sure n_steps is divisible by data
    n_steps = 100
    num_samples = int(len(X_intact_train) / n_steps)
    imputer = SAITS(n_steps=n_steps, n_features=num_features, n_layers=1, d_model=64, d_inner=32, n_head=4, d_k=64,
                    d_v=64, dropout=0.1, epochs=100, device="cuda:0")
    X_impute = X_missing_train.reshape(num_samples, n_steps, -1)
    X_impute[X_impute == 0] = np.nan
    imputer.fit(X_impute)

    #impute train set
    X_pred_train = imputer.impute(X_impute)
    X_pred_train = X_pred_train.reshape(num_samples * n_steps, -1)

    #impute test set
    num_samples = int(len(X_intact_test) / n_steps)
    impute_test = X_missing_test
    impute_test[impute_test == 0] = np.nan
    X_pred_test = imputer.impute(impute_test.reshape(num_samples, n_steps, -1))
    X_pred_test = X_pred_test.reshape(num_samples * n_steps, -1)

    # plot imputation
    plot_imputation(X_intact_train, X_pred_train, X_missing_train, index)

    return X_pred_train, X_pred_test

def train_evaluate_multiple_univariate_models(models:list, X_intact_train, X_intact_test, X_missing_train, X_missing_test, index):
    """
    Trains a subset of univariate baseline forecasting models
    """
    # X_impute_train, X_impute_test = train_saits_imputer(X_intact_train, X_intact_test, X_missing_train, X_missing_test, index)
    X_impute_train, X_impute_test = np.nan_to_num(X_intact_train, nan=0), np.nan_to_num(X_missing_test, nan=0)

    train_series = TimeSeries.from_times_and_values(times= index[:len(X_impute_train)],values=X_impute_train)
    for model in models:
        predictions = None
        for i in tqdm(train_series.columns):
            series = train_series.univariate_component(i)
            model.fit(series)
            if predictions is None:
                predictions = model.predict(len(X_intact_test))
            else:
                predictions = predictions.concatenate(model.predict(len(X_intact_test)), ignore_time_axis=True, axis=1)

        print(model)
        predictions = np.squeeze(predictions.all_values())
        compute_metrics(X_intact_test, predictions)

        plot_imputation(X_intact_test, predictions, X_missing_test, index)

def train_evaluate_multiple_multivariate_models(models:list, X_intact_train, X_intact_test, X_missing_train, X_missing_test, index):
    """
    trains a set of multivariate forecasting models
    """
    #switch between real interpolation and dummy interpolation
    X_impute_train, X_impute_test = train_saits_imputer(X_intact_train, X_intact_test, X_missing_train, X_missing_test, index)
    # X_impute_train, X_impute_test = np.nan_to_num(X_intact_train, nan=0), np.nan_to_num(X_missing_test, nan=0)

    train_series = TimeSeries.from_times_and_values(times=index[:len(X_impute_train)], values=X_impute_train)
    for model in models:
        model.fit(train_series)
        predictions = model.predict(len(X_missing_test))


        print(model)
        predictions = np.squeeze(predictions.all_values())
        compute_metrics(X_intact_test, predictions)

        plot_imputation(X_intact_test, predictions, X_missing_test, index)


def compute_metrics(target, forecast):
    """
    Computes forecasting metrics
    """


    try:
        r2 = r2_score(target, forecast)
    except ValueError:
        r2 = 1e+9

    target = np.nan_to_num(target, nan=0)
    indicating_mask = np.zeros_like(target)
    indicating_mask[np.nonzero(target)] = 1

    mae = cal_mae(forecast, target, indicating_mask)
    mse = cal_mse(forecast, target, indicating_mask)
    rmse = cal_rmse(forecast, target, indicating_mask)
    mre = cal_mre(forecast, target, indicating_mask)

    print(f"R2_score: {r2}, mae: {mae}, mse: {mse}, rmse: {rmse} mre: {mre}")



def train_proposed_model(X_intact:np.ndarray, X_missing:np.ndarray, resample_rate:str, index:pd.DatetimeIndex, columns:list):
    """
    Trains complex transformer architecture
    :param X_intact: complete data
    :param X_missing: data with missing values
    :param resample_rate: data frequency (hourly or daily)
    :param index: timesteps index
    :param columns: columns
    :return: trained model
    """

    forecast_window = 1
    if resample_rate == "h":
        forecast_window = 24

    num_features = X_intact.shape[1]
    "custom model"
    model = TransformerForecastImputeTrainer(input_chunk_length=7*forecast_window, decoder_length=4*forecast_window, output_chunk_length=1*forecast_window, num_features=num_features,
                                 n_epochs=10, #100
                                  batch_size=32,
                                  d_model= 64,
                                  dim_feedforward=512,
                                  num_layers=1,
                                imputation=True,
                               advanced_impute=True,
                               resample_rate= resample_rate,
                                diag_mask=True)

    model.fit_impute(X_missing, X_intact, index, missing_rate=missing_rate)

    return model

def evaluate_proposed_model(model, X_intact_test, X_missing_test, resample_rate:str, index:pd.DatetimeIndex):
    """
    evaluates complex architecture
    :param X_intact: complete data
    :param X_missing: data with missing values
    :param resample_rate: data frequency (hourly or daily)
    :param index: timesteps index
    :return: results
    """

    forecast_window = 1
    if resample_rate == "h":
        forecast_window = 24


    X_missing_test = np.nan_to_num(X_missing_test, nan=0)


    if model.advanced_impute:
        #examine quality of imputation
        imputes, imputes_targets, missing_mask = model.impute(X_missing_test, X_intact_test, index)
        original_sequence = deepcopy(imputes_targets)
        original_sequence[missing_mask] = np.nan
        plot_imputation(imputes_targets, imputes, original_sequence, index)

    #examine forecast
    forecast, target, missing_mask = model.predict_impute(X_missing_test, X_intact_test, time_index = index)



    r2 = r2_score(target, forecast)

    indicating_mask = np.zeros_like(target)
    indicating_mask[np.nonzero(target)] = 1

    mae = cal_mae(forecast, target, indicating_mask)
    mse = cal_mse(forecast, target, indicating_mask)
    rmse = cal_rmse(forecast, target, indicating_mask)
    mre = cal_mre(forecast, target, indicating_mask)


    print(f"R2_score: {r2}, mae: {mae}, mse: {mse}, rmse: {rmse} mre: {mre}")

    original_sequence = deepcopy(target)
    original_sequence[missing_mask] = np.nan

    plot_imputation(target, forecast, original_sequence, index)


def plot_imputation(data_y, forecast, data_to_impute, time_index:pd.DatetimeIndex):
    """
    plot the predicted imputation of a model
    :param data_y: the data with the true values
    :param forecast: prediction of missing values
    :param data_to_impute: original data with missing values
    :return: nice plot
    """


    data_to_impute[data_to_impute == 0] = np.nan
    data_y[data_y==0] = np.nan

    time_index = pd.DatetimeIndex(pd.Series(time_index).iloc[-len(data_to_impute):])



    fig = px.line()
    fig.add_scatter(x=time_index, y=data_y[:, 0], name="correct imputation", line=dict(color="green"))
    fig.add_scatter(x=time_index, y=forecast[:, 0], name="predicted imputation", line=dict(color="red"))
    fig.add_scatter(x=time_index, y=data_to_impute[:, 0], name="original sequence", line=dict(color="black"))
    fig.show()



if __name__ == '__main__':

    missing_rate = 0.2
    resample_rate = 'h'

    #select dataset
    sensor_data = forecast_impute_sensors(resample_rate=resample_rate)
    X_intact_train, X_intact_test, X_missing_train, X_missing_test, index = train_test_split(sensor_data, missing_rate)





    "train univariate baseline models"
    univariate_models = select_univariate_models()
    train_evaluate_multiple_univariate_models(univariate_models, X_intact_train, X_intact_test, X_missing_train, X_missing_test, index)

    "train multivariate baseline models"
    multivariate_models = select_multivariate_models(resample_rate)
    train_evaluate_multiple_multivariate_models(multivariate_models, X_intact_train, X_intact_test, X_missing_train, X_missing_test, index)

    "train proposed architecture"
    model = train_proposed_model(X_intact_train, X_missing_train, resample_rate, sensor_data.index, sensor_data.columns)
    evaluate_proposed_model(model, X_intact_test, X_missing_test, resample_rate, sensor_data.index[len(X_missing_train):])






