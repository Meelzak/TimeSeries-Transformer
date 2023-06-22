import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pypots.data import mcar, masked_fill

from datasets.impute_dataset import create_missing_chunks, create_variable_missing_chunks, compute_nan_ratio

"""
This file contains everything for loading the sensor data
It also divides the data into train, val and test sets
The data is also normalised so it can be directly fitted into the model.
"""
def _load_sensor_csv():
    """
    loads the sensor and weather datasets from csv
    Converts all columns to correct format
    :return: sensor and weather dataframes
    """
    sensor_data = pd.read_csv("../all_sensor_data_imputed.csv")
    weather_data = pd.read_csv("../all_weather_data.csv")

    sensor_data['timestamp'] = pd.to_datetime(sensor_data['timestamp'])
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])

    sensor_data = sensor_data.set_index('timestamp', drop=False)
    weather_data = weather_data.set_index('timestamp', drop=False)

    sensor_data.sort_index(inplace = True)
    weather_data.sort_index(inplace = True)

    sensor_data = sensor_data.drop_duplicates()
    weather_data = weather_data.drop_duplicates()

    return sensor_data, weather_data

def _multiple_sensor_temperature(sensor_data: pd.DataFrame)->pd.DataFrame:
    """
    Create a transposed table with the sensor als columns and the temperature as values
    :param sensor_data: the original dataset
    :return: transposed dataset
    """
    sensor_list = sensor_data['devId'].unique()

    sensor_temperatures = {}

    for i in range(len(sensor_list)):
        current_sensor = sensor_data[sensor_data['devId'] == sensor_list[i]]
        current_sensor_amis = current_sensor[current_sensor['appId']=='amis-kantoorruimtes']
        current_sensor_mediaan = current_sensor[current_sensor['appId']=='mediaan-kantoorruimtes']

        if len(current_sensor_amis)>0:
            sensor_temperatures[str(sensor_list[i])+'_amis'] = current_sensor_amis['temperature']
        if len(current_sensor_mediaan)>0:
            sensor_temperatures[str(sensor_list[i])+'_mediaan'] = current_sensor_mediaan['temperature']

        # sensor_temperatures[str(sensor_list[i]) + "_humidity"] = current_sensor['humidity']
        # sensor_temperatures[str(sensor_list[i]) + "_motion"] = current_sensor['motion']

    sensor_temperatures_df = pd.DataFrame(sensor_temperatures)

    mediaan = sensor_temperatures_df[['ers-10_mediaan',
                             'ers-11_mediaan',
                             'ers-12_mediaan',
                             'ers-13_mediaan',
                             'ers-14_mediaan',
                             'ers-15_mediaan',
                             'ers-16_mediaan',
                             'ers-17_mediaan',
                             'ers-18_mediaan',
                             'ers-19_mediaan',
                             'ers-20_mediaan',
                             'ers-21_mediaan',
                             'ers-22_mediaan',
                             'ers-23_mediaan',
                             'ers-24_mediaan',
                             'ers-4_mediaan',
                             'ers-5_mediaan',
                             'ers-6_mediaan',
                             'ers-7_mediaan',
                             'ers-8_mediaan',
                             'ers-9_mediaan']]
    mediaan = mediaan[mediaan.index > pd.Timestamp('2022-09-04')]

    amis = sensor_temperatures_df[['erseye-1_amis',
                                   'ers-3_amis',
                                   'ers-1_amis',
                                   'ers-2_amis',
                                   'erseye-2_amis',
                                   'ers-4_amis',
                                   'ers-5_amis',
                                   'ers-6_amis',
                                   ]]


    return mediaan
    # return amis


def no_nan_sensors(resample_rate = 'd'):
    """
    Return the subset of sensors with almost no missing values
    The remaining nan values are imputed by the median
    :param resample_rate: (string) do we want daily or hourly values
    :return: subset dataframe
    """
    sensor_data , weather_data = _load_sensor_csv()
    df = _multiple_sensor_temperature(sensor_data)

    df = df.resample(resample_rate).mean()

    nan_columns = df.isna().sum()

    df = df.fillna(df.median().median())

    return df


def forecast_impute_sensors(resample_rate = 'd'):
    """
    returns a dataset used for training the impute and forecast transformer
    :param resample_rate: (string) do we want daily or hourly values
    :return: subset dataframe
    """
    sensor_data , weather_data = _load_sensor_csv()
    df = _multiple_sensor_temperature(sensor_data)

    df = df.resample(resample_rate).mean()

    return df



"""
----------------------------------------------------------------------------------------------------------------------
These methods below are only used for the imputation models
These are not compatible with the forecasting and forecasting_impute models
"""

def _prepare_sensor_data(sensor_data: pd.DataFrame, n_steps=48, freq='d', missing_rate=0.5)->np.array:
    """
    Prepare data such that it can be imputed in the model
    Currently it only uses information of a single sensor
    :param sensor_data: sensor dataframe
    :param sensor_name: Name of the sensor we would like to predict
    :param n_steps: number of steps used by the transformer for prediction
    if n_steps=48 the last 48 hours to make a prediction
    :param freq: if we want daily or hourly forecasts
    :param missing_rate: missing rate to artifically generate missing values
    :return: train, val, test data
    """

    sensor_temperatures = _multiple_sensor_temperature(sensor_data)
    sensor_temperatures = sensor_temperatures.resample(freq).mean()


    num_samples = int(len(sensor_temperatures) / n_steps)
    sensor_amount = len(sensor_temperatures.columns)

    sensor_temperatures = sensor_temperatures.head(num_samples*n_steps)

    scaler =  StandardScaler() #MinMaxScaler(feature_range=(-1,1)) #StandardScaler()
    X = scaler.fit_transform(sensor_temperatures.to_numpy())
    X = X[:(num_samples*n_steps)]
    X_reshape = X.reshape(num_samples, n_steps, -1)

    X_train = X_reshape[:len(X_reshape)-int(num_samples*0.25)]
    X_val = X_reshape[len(X_reshape)-int(num_samples*0.25):]

    ## create a test set with missing values (artifically remove 10 percent of datapoints for validation
    # X_intact, X_missing, missing_mask, indicating_mask = mcar(X_val, missing_rate)
    mean, std = compute_nan_ratio(X)
    X_intact, X_missing, missing_mask, indicating_mask = create_variable_missing_chunks(X_val, missing_rate)

    artficial_rate = np.count_nonzero(X_missing)
    real_rate = np.count_nonzero(X_intact)
    print("artificial missing length: ", artficial_rate)
    print("original missing length: ", real_rate)




    X_missing = masked_fill(X_missing, 1 - missing_mask, np.nan)
    X_test_complete = X_intact
    X_test_predict = X_missing
    X_test_indicating_mask = indicating_mask

    return X_train, X_test_complete, X_test_predict, X_test_indicating_mask, sensor_temperatures.index

def load_sensor_data(missing_rate: float, n_steps=48, freq='d'):
    """
    creates sensor dataset directly suitable for training imputation
    It creates hourly timesteps
    :param sensor_name: which sensor you want to train the model (can be all)
    :param n_steps: how many steps (hours) you want to use as forecasting window (MUST CORRESPOND TO FREQUENCY)
    :param freq: if we want daily or hourly forecasts (d or h)
    :param missing_rate: missing rate to artifically generate missing values
    :return: train, val, test sets
    """
    sensor_data, weather_data = _load_sensor_csv()

    X_train, X_test_complete, X_test_predict, X_test_indicating_mask, time_index = _prepare_sensor_data(sensor_data, n_steps=n_steps, freq=freq, missing_rate=missing_rate)

    return X_train, X_test_complete, X_test_predict, X_test_indicating_mask, time_index