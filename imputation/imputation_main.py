import pandas as pd
import numpy as np
import plotly.express as px
from pypots.data import mcar, masked_fill
from pypots.imputation import SAITS, LOCF, BRITS, Transformer
from pypots.utils.metrics import cal_mae, cal_rmse, cal_mre, cal_mse

from datasets.impute_dataset import create_missing_chunks, compute_nan_ratio, create_variable_missing_chunks
from imputation.models.median_imputer import MedianImputer
from imputation.models.spline_interpolation import SplineInterpolation
from imputation.models.fancy_impute_interpolation_models import KNN_imputation, MatrixFactorizationImputation, SingularValueDecomposition

from datasets.sensor_data_loading import load_sensor_data
from imputation.models.transformer.transformer_imputer_trainer import TransformerImputerTrainer

"""
This class experiments with several imputation models and strategies
Models are tested on the sensor data but also other benchmark datasets
@Author MeelsL
"""

def train_advanced_imputer(X_train, time_index:pd.DatetimeIndex, n_steps=48, epochs=100, freq = 'd', missing_rate=0.5):
    """
    Train imputation model of the proposed paper architecture
    Because the input looks a bit different I created a different method
    :param X_train: the training data with missing values
    :param X_test_complete: training data with correct values
    :param time_index: the time index according to training data
    :param n_steps: number of steps to take into account each prediction
    :param freq: if we want daily or hourly forecast
    :param missing_rate: missing rate to artifically generate missing values
    :return: trained model
    """

    forecast_window = 1
    if freq == 'h':
        forecast_window = 24

    X_intact, X_missing, missing_mask, indicating_mask = mcar(X_train, missing_rate)

    num_features = X_train.shape[2]
    X_missing = X_missing.reshape(-1, num_features)
    X_intact = X_intact.reshape(-1, num_features)

    model = TransformerImputerTrainer(input_chunk_length=n_steps, decoder_length=5*forecast_window, output_chunk_length=1*forecast_window, num_features=num_features,
                                 n_epochs=epochs,
                                  batch_size=32,
                                  d_model= 64,
                                  dim_feedforward=512,
                                  num_layers=1,
                                imputation=True,
                               advanced_impute=False,
                               resample_rate=freq,
                                diag_mask=False)

    model.fit_impute(X_missing, X_intact, time_index, missing_rate=missing_rate)

    return model

def evaluate_advanced_imputer(model: TransformerImputerTrainer, X_test_complete, X_test_predict, X_test_indicating_mask, time_index:pd.DatetimeIndex):
    """
    evaluate trained model based on the proposed architecture
    :param model: trained model
    :param X_test_complete: complete dataset with no missing values
    :param X_test_predict: dataset with missing values
    :param X_test_indicating_mask: indicator if value was missing in training set or not
    :param time_index: time_index of the given datapoints
    :return: prediction
    """

    original_shape = X_test_complete.shape

    num_features = X_test_predict.shape[2]
    X_test_predict = X_test_predict.reshape(-1, num_features)
    X_test_predict = np.nan_to_num(X_test_predict, nan=0)
    X_test_complete = X_test_complete.reshape(-1, num_features)
    X_test_indicating_mask = X_test_indicating_mask.reshape(-1, num_features)

    X_pred, X_correct, _ = model.impute(X_test_predict, X_test_complete, time_index)

    X_test_indicating_mask = X_test_indicating_mask[:-model.input_chunk_length]
    X_test_predict = X_test_predict[:-model.input_chunk_length]


    mae = cal_mae(X_pred, X_correct, X_test_indicating_mask)
    mse = cal_mse(X_pred, X_correct, X_test_indicating_mask)
    rmse = cal_rmse(X_pred, X_correct, X_test_indicating_mask)
    mre = cal_mre(X_pred, X_correct, X_test_indicating_mask)
    print("mae: ", mae, " mse: ", mse, " rmse: ", rmse, " mre: ", mre)

    new_shape = (original_shape[0]-1, original_shape[1], original_shape[2])

    return X_correct.reshape(new_shape), X_test_predict.reshape(new_shape), X_pred.reshape(new_shape)


def train_multiple_models(X_train, X_val, n_steps:int, epochs:int)->list:
    """
    train multiple imputation model
    Different models possible depending on the commented lines
    :param X_train: the training data
    :param X_val: the validation data
    :param n_steps: the forecast window
    :param epochs: number of epochs to train the model
    :return: list of multiple models for evaluation
    """

    # model = MODIFIED_SAITS(n_steps=n_steps, n_features=np.shape(X_train)[2], n_layers=2, d_model=64, d_inner=32, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=epochs, device="cuda:0")
    #
    # brits = BRITS(n_steps=n_steps, n_features=np.shape(X_train)[2], rnn_hidden_size=256, epochs=epochs, device="cuda:0")
    # brits.fit(X_train, X_val)
    #
    # transformer = Transformer(n_steps=n_steps, n_features=np.shape(X_train)[2], n_layers=2, d_model=256, d_inner=128,
    #                        n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=epochs, device="cuda:0")
    # transformer.fit(X_train, X_val)
    #
    # saits = SAITS(n_steps=n_steps, n_features=np.shape(X_train)[2], n_layers=2, d_model=256, d_inner=128,
    #                        n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=epochs, device="cuda:0", batch_size=128)
    # saits.fit(X_train, X_val)
    #
    # # return [saits]
    # return [brits, transformer, saits]



    median = MedianImputer().fit(X_train)
    linear = SplineInterpolation(method="linear").fit(X_train)
    spline = SplineInterpolation(method="spline").fit(X_train)
    svd = SingularValueDecomposition().fit(X_train)
    mf = MatrixFactorizationImputation().fit(X_train)
    knn = KNN_imputation(k=5).fit(X_train)
    return [median, linear, spline, knn, svd, mf]



def evaluate(model, X_test_complete, X_test_predict, X_test_indicating_mask):
    """
    evaluate trained model
    :param model: trained model
    :param X_test_complete: complete dataset with no missing values
    :param X_test_predict: dataset with missing values
    :param X_test_indicating_mask: indicator if value was missing in training set or not
    :return: prediction
    """

    X_pred = model.impute(X_test_predict)


    mae = cal_mae(X_pred, X_test_complete, X_test_indicating_mask)
    mse = cal_mse(X_pred, X_test_complete, X_test_indicating_mask)
    rmse = cal_rmse(X_pred, X_test_complete, X_test_indicating_mask)
    mre = cal_mre(X_pred, X_test_complete, X_test_indicating_mask)
    print("mae: ", mae, " mse: ", mse, " rmse: ", rmse, " mre: ", mre)

    return X_pred



def plot_imputation(X_test_complete, X_test_missing, X_test_predict, time_index):
    """
    plot the predicted imputation of a model
    :param X_test_complete: the data with the true values
    :param X_test_missing: data with missing values
    :param X_test_predict: prediction of missing values
    :param time_index: time index used for the x-axis
    :return: nice plot
    """

    series, seq_len, features = X_test_missing.shape

    X_test_missing = X_test_missing.reshape(series * seq_len, features)
    X_test_missing[X_test_missing==0] = np.nan
    X_test_complete = X_test_complete.reshape(series * seq_len, features)
    X_test_predict = X_test_predict.reshape(series * seq_len, features)



    time_index = time_index[-len(X_test_predict):]

    fig = px.line()
    fig.add_scatter(x=time_index, y=X_test_complete[:, 0], name="correct imputation", line=dict(color="green"))
    fig.add_scatter(x=time_index, y=X_test_predict[:, 0], name="predicted imputation", line=dict(color="red"))
    fig.add_scatter(x=time_index, y=X_test_missing[:, 0], name="original sequence", line=dict(color="black"))
    fig.show()


if __name__ == '__main__':
    # decide which dataset to laod
    resample_rate = 'h'
    missing_rate = 0.2
    n_steps = 7
    if resample_rate == 'h':
        n_steps = n_steps*24

    "specify which dataset to load"
    X_train, X_test_complete, X_test_predict, X_test_indicating_mask, time_index = load_sensor_data(n_steps=n_steps,freq=resample_rate, missing_rate=missing_rate)
    # X_train, X_val, X_test_complete, X_test_predict, X_test_indicating_mask, n_steps = load_air_quality_dataset(missing_rate)




    "advanced transformer model"
    model = train_advanced_imputer(X_train, time_index, n_steps=n_steps, epochs=10, freq=resample_rate, missing_rate=missing_rate)
    X_test_complete, X_test_predict, X_pred = evaluate_advanced_imputer(model, X_test_complete, X_test_predict, X_test_indicating_mask, time_index)
    plot_imputation(X_test_complete, X_test_predict, X_pred, time_index)
