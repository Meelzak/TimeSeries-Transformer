import numpy as np
import pandas as pd
import torch
from pypots.data import masked_fill, mcar
from torch.utils.data import Dataset

from datasets.generate_time_features import time_features
from datasets.impute_dataset import create_missing_chunks, create_variable_missing_chunks


class Dataset_from_arrays(Dataset):
    """
    This class creates a pytorch dataset for missing data
    :param data_x is the missing data
    :param data_y is the complete data
    Dataset used in test mode for the impute and forecast task
    Requires the missing data and imputed data in array format
    @Author MeelsL
    """

    def __init__(self, data_x, data_y, input_chunk_length:int, decoder_length:int, output_chunk_length:int, imputation: bool,
                 missing_rate:float,
                 freq='d', time_index: pd.DatetimeIndex = None, parent=True):

        self.seq_len = input_chunk_length
        self.label_len = decoder_length
        self.pred_len = output_chunk_length



        self.time_index = time_index
        self.freq = freq
        self.imputation = imputation
        self.missing_rate = missing_rate

        if parent:
            self.__generate_from_test_set__(data_x, data_y)

    def __generate_from_test_set__(self, data_x, data_y):
        data_stamp = time_features(self.time_index, freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        missing_mask = torch.zeros(data_x.shape)
        missing_mask[data_x == 0] = 1

        self.data_x = data_x
        self.data_y = data_y
        self.data_stamp = data_stamp
        self.missing_mask = missing_mask

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]


        "Select the input and output window to be y"
        seq_y = self.data_y[s_begin:r_end]
        seq_y_mark = self.data_stamp[s_begin:r_end]
        seq_y_missing = self.missing_mask[s_begin:r_end]



        if self.imputation and self.missing_rate != 0:
            seq_y[seq_y == 0] = np.nan
            "ïf we always want to randomly mask input"
            seq_y, seq_x, missing_mask, seq_y_missing = mcar(seq_y, rate=self.missing_rate)
            seq_x = seq_x[:self.seq_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_missing

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

"""
This class creates a time series dataset from a pandas dataframe.
It will generate datetime features as well (so weekday boolean, day of month, day of week, hour of day etc.)
These additional features are helpful for the model to learn seasonality features

Inherits from Dataset_from_arrays to reduce duplicated methods
@Author MeelsL
"""
class Dataset_Pandas(Dataset_from_arrays):
    def __init__(self, df:pd.DataFrame, input_chunk_length:int, decoder_length:int, output_chunk_length:int,
                 imputation: bool, missing_rate:float,
                 freq='d', time_index: pd.DatetimeIndex = None):

        super().__init__(df, df, input_chunk_length, decoder_length, output_chunk_length,
                 imputation, missing_rate,
                 freq, time_index, parent=False)

        self.df_raw = df

        self.__read_data__()

    def __read_data__(self):
        """
        read normal dataset
        :return: transformed dataset
        """
        df_raw = self.df_raw


        if isinstance(df_raw, pd.DataFrame):
            data = df_raw.values
        else:
            data = df_raw


        if self.time_index is None:
            data_stamp = time_features(df_raw.index, freq=self.freq)
        else:
            data_stamp = time_features(self.time_index, freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

        indicating_mask = torch.zeros(data.shape)
        indicating_mask[data == 0] = 1

        self.missing_mask = indicating_mask


"""
data masking operators
"""
class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu", attn_mask=None):
        if attn_mask is None:
            _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        else:
            _mask = attn_mask.reshape((scores.shape[-1], scores.shape[-1]))
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
