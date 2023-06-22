from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from forecasting.models.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from forecasting.models.layers.SelfAttention_Family import FullAttention, AttentionLayer, ProbAttention, \
    AutoCorrelationLayer, AutoCorrelation
from forecasting.models.layers.FourierTransformations import FourierBlock, FourierCrossAttention
from forecasting.models.layers.Embed import DataEmbedding, DataEmbedding_wo_pos

from forecasting.models.layers.FedFormer_EncDec import Encoder as FedEncoder, Decoder as FedDecoder, \
    EncoderLayer as FedEncoderLayer, DecoderLayer as FedDecoderLayer, my_Layernorm, series_decomp


class AdvancedTransformer(nn.Module):
    """
    Transformer torch module
    Transformer contains several variations for imputation, forecast and impute_forecast
    @Author MeelsL
    """

    def __init__(self, device:torch.device, num_features = 1, input_chunk_length = 96, decoder_length = 48, output_chunk_length = 96, d_model=64, dim_feedforward=512
                 , num_layers=1,dropout=0.1, n_heads=4, task_name = "long_term_forecast", advanced_impute=False,
                 resample_rate='h', diag_mask=True):
        super(AdvancedTransformer, self).__init__()
        self.task_name = task_name
        self.label_len = decoder_length
        self.input_chunk_length = input_chunk_length
        self.pred_len = output_chunk_length
        self.output_attention = False
        self.device = device
        self.advanced_impute = advanced_impute
        self.resample_rate = resample_rate
        self.diag_mask = diag_mask

        if self.task_name == "imputation":
            self.seq_len = input_chunk_length
        else:
            self.seq_len = input_chunk_length + output_chunk_length

        if self.task_name == "impute_and_forecast" and self.advanced_impute:
            num_features_input = num_features #*2
            num_features_output = num_features
        else:
            num_features_input = num_features
            num_features_output = num_features

        # Embedding
        self.enc_embedding = DataEmbedding(num_features_input, d_model, 'timeF', self.resample_rate,dropout)
        # self.enc_embedding = DataEmbedding_wo_pos(num_features_input, d_model, 'timeF', self.resample_rate,dropout)


        if self.task_name == "impute_and_forecast" or self.task_name == "imputation":
            "choose which attention type to use"
            self.impute_encoder = self.regular_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask=diag_mask)
            # self.impute_encoder = self.probsparse_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask=diag_mask, imputation=True)
            # self.impute_encoder = self.fedformer_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers)
            # self.impute_encoder = self.autoformer_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers)

            if self.task_name == "imputation" or self.task_name == "impute_and_forecast":
                self.impute_layer = nn.Linear(d_model, num_features_output, bias=True)

            if self.task_name == "impute_and_forecast":
                self.forecast_embedding = DataEmbedding(num_features_input, d_model, 'timeF', self.resample_rate, dropout)
                # self.forecast_embedding = DataEmbedding_wo_pos(num_features_input, d_model, 'timeF', self.resample_rate, dropout)

        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'impute_and_forecast':
            # Encoder
            "choose which attention type to use"
            # self.forecast_encoder = self.regular_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask=False)
            self.forecast_encoder = self.probsparse_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask=False, imputation=False)
            # self.forecast_encoder = self.fedformer_attention_encoder(d_model, dim_feedforward, n_heads, dropout,num_layers)
            # self.forecast_encoder = self.autoformer_attention_encoder(d_model, dim_feedforward, n_heads, dropout,num_layers)


            self.dec_embedding = DataEmbedding(num_features_output, d_model, 'timeF', self.resample_rate, dropout)
            # self.dec_embedding = DataEmbedding_wo_pos(num_features_input, d_model, 'timeF', self.resample_rate, dropout)

            "choose which attention type to use"
            # self.decoder = self.regular_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output)
            self.decoder = self.probsparse_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output)
            # self.decoder, self.decomp = self.fedformer_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output)
            # self.decoder, self.decomp = self.autoformer_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        model forecast
        """
        # attn_mask_enc = torch.eye(x_enc.size(dim=1)).to(x_enc.device)
        # attn_mask_enc = attn_mask_enc.unsqueeze(0).unsqueeze(1)
        attn_mask_enc = None


        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.forecast_encoder(enc_out, attn_mask=attn_mask_enc)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def fedformer_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        forecast for fedformer (uses series decompositon)
        """
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input

        if trend_init[:, -(self.label_len+self.pred_len):, :].size(dim=1)!=x_mark_dec.size(dim=1):
            #if predict
            trend_init = trend_init[:, max(0, self.input_chunk_length - x_dec.size()[1] + self.pred_len):self.input_chunk_length + self.pred_len, :]
            trend_init = torch.cat([trend_init, mean], dim=1)
        else:
            #if train
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)


        if seasonal_init[:, -(self.label_len+self.pred_len):, :].size(dim=1) != x_mark_dec.size(dim=1):
            #if predict
            seasonal_init = seasonal_init[:, max(0, self.input_chunk_length-x_dec.size()[1]+self.pred_len):self.input_chunk_length+self.pred_len, :]
            seasonal_init =  F.pad(seasonal_init, (0, 0, 0, self.pred_len))
        else:
            #if train
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))


        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, attns = self.forecast_encoder(enc_out, attn_mask=None)
        # dec
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def autoformer_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        forecast using autoformer (using series decomposition)
        """
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)


        if trend_init[:, -(self.label_len+self.pred_len):, :].size(dim=1)!=x_mark_dec.size(dim=1):
            #if predict
            trend_init = trend_init[:, max(0, self.input_chunk_length - x_dec.size()[1] + self.pred_len):self.input_chunk_length + self.pred_len, :]
            trend_init = torch.cat([trend_init, mean], dim=1)
        else:
            #if train
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)


        if seasonal_init[:, -(self.label_len+self.pred_len):, :].size(dim=1) != x_mark_dec.size(dim=1):
            #if predict
            seasonal_init = seasonal_init[:, max(0, self.input_chunk_length-x_dec.size()[1]+self.pred_len):self.input_chunk_length+self.pred_len, :]
            seasonal_init =  torch.cat([seasonal_init, zeros], dim=1)
        else:
            #if train
            seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.forecast_encoder(enc_out, attn_mask=None)

        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)

        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc):

        if self.diag_mask:
            attn_mask = torch.eye(x_enc.size(dim=1)).to(x_enc.device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        else:
            attn_mask=None

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.impute_encoder(enc_out, attn_mask=attn_mask)

        dec_out = self.impute_layer(enc_out)
        return dec_out

    def impute_advanced_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, missing_mask):
        """
        advanced imputation and forecasting
        First use the imputation encoder to impute data (Imputation is at timestep t_i)
        Then use the forecasting encoder to forecast (Forecast is at timestep t_i+1)
        """

        if self.diag_mask:
            attn_mask = torch.eye(x_enc.size(dim=1)).to(x_enc.device)
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(1)
        else:
            attn_mask=None

        "first imputation"
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        impute_out, attns = self.impute_encoder(enc_out, attn_mask = attn_mask)
        imputation = self.impute_layer(impute_out)

        "then forecasting"
        forecast_enc, attns = self.forecast_encoder(impute_out, attn_mask = None) #enc_forc_out
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        forecast = self.decoder(dec_out, forecast_enc, x_mask=None, cross_mask=None)

        # forecast = self.autoformer_forecast(forecast_input, x_mark_enc, x_dec, x_mark_dec)
        # forecast = self.fedformer_forecast(forecast_input, x_mark_enc, x_dec, x_mark_dec)

        return forecast, imputation

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            if isinstance(self.decoder,FedDecoder):
                if isinstance(self.enc_embedding, DataEmbedding_wo_pos):
                    dec_out = self.autoformer_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                else:
                    dec_out = self.fedformer_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out, None
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc)
            return None, dec_out  # [B, L, D]
        if self.task_name == "impute_and_forecast":

            if self.advanced_impute:
                missing_mask = torch.ones(x_enc.size()).to(self.device)
                missing_mask[x_enc == 0] = 0
                # input = torch.cat([x_enc, missing_mask], dim=2)

                input = x_enc

                forecast, imputation = self.impute_advanced_forecast(input, x_mark_enc, x_dec, x_mark_dec, missing_mask)
                return forecast, imputation
            else:
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out, None

        return None

    @staticmethod
    def regular_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask:bool):
        """
        returns standard attention encoder layer
        """
        print("Standard Encoder Initialized")
        return Encoder(

            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(diag_mask, 3, attention_dropout=dropout,
                                      output_attention=False), d_model, n_heads),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(num_layers)
            ],

            norm_layer=torch.nn.LayerNorm(d_model)
        )

    @staticmethod
    def regular_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output):
        """
        returns standard attention decoder layer
        """
        print("Standard Decoder Initialized")
        return Decoder(

            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, 3, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        FullAttention(False, 3, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                )
                for l in range(num_layers)
            ],

                norm_layer=torch.nn.LayerNorm(d_model),
                projection=nn.Linear(d_model, num_features_output, bias=True)
            )
    @staticmethod
    def probsparse_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers, diag_mask:bool, imputation:bool):
        """
        Convolution layer is not compatible with imputation
        returns probsparse attention encoder layer
        """
        print("Probsparse Encoder Initialized")
        if imputation:
            return Encoder(

                [
                    EncoderLayer(
                        AttentionLayer(
                            ProbAttention(diag_mask, factor=3, attention_dropout=dropout,
                                          output_attention=False),
                            d_model, n_heads),
                        d_model,
                        dim_feedforward,
                        dropout=dropout,
                        activation='gelu'
                    ) for l in range(num_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
            )
        else:
            return Encoder(

                [
                    EncoderLayer(
                        AttentionLayer(
                            ProbAttention(diag_mask, factor=3, attention_dropout=dropout,
                                          output_attention=False),
                            d_model, n_heads),
                        d_model,
                        dim_feedforward,
                        dropout=dropout,
                        activation='gelu'
                    ) for l in range(num_layers)
                ], [
                    ConvLayer(
                        d_model
                    ) for l in range(num_layers)
                ],

            norm_layer=torch.nn.LayerNorm(d_model)
            )


    @staticmethod
    def probsparse_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output):
        """
        returns probsparse attention decoder layer
        """
        print("Probsparse Decoder Initialized")
        return Decoder(

            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, factor=3, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    AttentionLayer(
                        ProbAttention(False, factor=3, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation='gelu',
                )
                for l in range(num_layers)
            ],

            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, num_features_output, bias=True)
        )


    def fedformer_attention_encoder(self, d_model, dim_feedforward, n_heads, dropout, num_layers):
        """
        returns fedformer encoder
        """
        print("Fedformer Encoder Initialized")
        return FedEncoder(
            [
                FedEncoderLayer(
                    AutoCorrelationLayer(
                        FourierBlock(in_channels=d_model,
                                     out_channels=d_model,
                                     seq_len=self.seq_len,
                                     modes=32,
                                     mode_select_method='random'),

                        # MultiWaveletTransform(ich=d_model, L=1, base='legendre'),

                        d_model, n_heads),
                    d_model,
                    dim_feedforward,
                    moving_avg=25,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

    def fedformer_attention_decoder(self, d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output):
        """
        returns fedformer decoder
        """
        print("Fedformer Decoder Initialized")
        return FedDecoder(
            [
                FedDecoderLayer(
                    AutoCorrelationLayer(
                        FourierBlock(in_channels=d_model,
                                     out_channels=d_model,
                                     seq_len=self.seq_len // 2 + self.pred_len,
                                     modes=32,
                                     mode_select_method='random'),

                        # MultiWaveletTransform(ich=d_model, L=1, base='legendre'),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        FourierCrossAttention(in_channels=d_model,
                                              out_channels=d_model,
                                              seq_len_q=self.seq_len // 2 + self.pred_len,
                                              seq_len_kv=self.seq_len,
                                              modes=32,
                                              mode_select_method='random')

                        # MultiWaveletCross(in_channels=d_model,
                        #                                     out_channels=d_model,
                        #                                     seq_len_q=self.seq_len // 2 + self.pred_len,
                        #                                     seq_len_kv=self.seq_len,
                        #                                     modes=32,
                        #                                     ich=d_model,
                        #                                     base='legendre',
                        #                                     activation='tanh')
                        ,
                        d_model, n_heads),
                    d_model,
                    num_features_output,
                    dim_feedforward,
                    moving_avg=25,
                    dropout=dropout,
                    activation='gelu',
                )
                for l in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, num_features_output, bias=True)
        ),  series_decomp(25)

    @staticmethod
    def autoformer_attention_encoder(d_model, dim_feedforward, n_heads, dropout, num_layers):
        print("Autoformer Encoder Initialized")
        return FedEncoder(
            [
                FedEncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor=3, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    dim_feedforward,
                    moving_avg=25,
                    dropout=dropout,
                    activation="gelu"
                ) for l in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model)
        )

    @staticmethod
    def autoformer_attention_decoder(d_model, dim_feedforward, n_heads, dropout, num_layers, num_features_output):
        print("Autoformer Decoder Initialized")
        return FedDecoder(
            [
                FedDecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor=3, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor=3, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads),
                    d_model,
                    num_features_output,
                    dim_feedforward,
                    moving_avg=25,
                    dropout=dropout,
                    activation="gelu",
                )
                for l in range(num_layers)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, num_features_output, bias=True)
        ),  series_decomp(25)

    def get_imputation_parameters(self):
        """
        :return: all imputation layers
        """
        return list(self.enc_embedding.parameters()) + list(self.impute_encoder.parameters()) + list(self.impute_layer.parameters())

    def get_forecast_parameters(self):
        """
        :return: all forecasting layers
        """
        if self.advanced_impute and self.task_name == "impute_and_forecast":
            return list(self.forecast_embedding.parameters()) + list(self.forecast_encoder.parameters()) + list(self.dec_embedding.parameters()) + list(self.decoder.parameters())
        if self.task_name == "long_term_forecast":
            return list(self.enc_embedding.parameters()) + list(self.forecast_encoder.parameters()) + list(self.dec_embedding.parameters()) + list(self.decoder.parameters())

