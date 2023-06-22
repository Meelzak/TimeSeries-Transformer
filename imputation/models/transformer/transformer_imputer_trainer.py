import time

import pandas as pd
import torch
from pypots.utils.metrics import cal_mse
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.advanced_loader import Dataset_from_arrays
from forecast_impute.models.TransformersBase import TransformersBase
from forecast_impute.models.transformer_module import AdvancedTransformer


class TransformerImputerTrainer(TransformersBase):

    """
    This class trains the transformer of the AdvancedTransformer class
    This is specifically for imputation only
    It contains training and prediction loops for imputation only
    @Author MeelsL
    """

    def __init__(self, input_chunk_length: int, decoder_length: int, output_chunk_length: int, resample_rate:str, num_features=1,
                 batch_size: int = 32, n_epochs=100, d_model=64, dim_feedforward=512, num_layers=1, dropout=0.1,
                 imputation=True, advanced_impute=True, diag_mask=True):


        super().__init__(input_chunk_length, decoder_length, output_chunk_length, resample_rate, num_features,
                 batch_size, n_epochs, d_model, dim_feedforward, num_layers, dropout, imputation, advanced_impute)

        if self.advanced_impute:
            task = "impute_and_forecast"
        else:
            task = "imputation"


        self.model = AdvancedTransformer(num_features = num_features, input_chunk_length=input_chunk_length, decoder_length=decoder_length,
                                         output_chunk_length=output_chunk_length,
                                         d_model= d_model, dim_feedforward= dim_feedforward,
                                         num_layers = num_layers, dropout= dropout,
                                         task_name=task, device=self.device, advanced_impute = advanced_impute, resample_rate=resample_rate, diag_mask=diag_mask)

        self.model = self.model.to(self.device)


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

        if self.advanced_impute:
            return self.fit_loop(train_data, train_loader)
        else:
            return self.fit_simple(train_data, train_loader)


    def fit_simple(self, train_data, train_loader):
        """
        train the model only for imputation
        (Does not train the advanced architecture also suitable for forecasting
        :param train_data: training data
        :param train_loader: data loader for the batches
        :return: trained model
        """

        val_data, val_loader = train_data, train_loader

        lr = 0.005

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        criterion = cal_mse

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            iter_count = 0

            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_missing) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                seq_y_missing = seq_y_missing.bool().to(self.device)


                # model forward
                _, imputes = self.model(batch_x, batch_x_mark, None, None)

                loss = criterion(imputes, batch_y[:, :self.input_chunk_length, :]
                                              ,seq_y_missing[:, :self.input_chunk_length, :])

                loss.backward(retain_graph=False)
                optimizer.step()

                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | imputation_loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    iter_count = 0
                    time_now = time.time()

                # self.evaluate(val_loader, forecast_criterion)

            scheduler.step()

        return self.model

    def impute(self, data_to_impute, data_y, time_index:pd.DatetimeIndex):
        """
        Prediction of the imputation part of the advanced transformer architecture
        This method only tests the imputation part of the model hence no forecasting is involved.
        :param data_to_impute: data to impute
        :param data_y: correct imputed values
        :param time_index: time index of the data
        :return: imputed predictions by the model
        """

        n_steps = len(data_y) - self.input_chunk_length
        divider = int(n_steps / self.input_chunk_length)
        n_steps = divider * self.input_chunk_length


        input_data = Dataset_from_arrays(data_x=data_to_impute, data_y=data_y,
                                         input_chunk_length=self.input_chunk_length,
                                         decoder_length=self.decoder_length,
                                         output_chunk_length=self.output_chunk_length,
                                         freq=self.resample_rate, imputation=self.imputation, time_index=time_index,
                                         missing_rate=0)

        input_loader = DataLoader(input_data, batch_size=1, shuffle=False, num_workers=0)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_missing) in enumerate(input_loader):

                #only predict for each timestep
                if i % self.input_chunk_length == 0:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    seq_y_missing = seq_y_missing.bool().to(self.device)

                    batch_y_dec = batch_y.clone()
                    batch_y_dec[seq_y_missing] = 0

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y_dec[:, -self.output_chunk_length:, :]).float()


                    if i==0:
                        dec_inp = torch.cat([batch_y_dec[:, self.input_chunk_length-self.decoder_length:self.input_chunk_length, :], #:-self.output_chunk_length
                                             dec_inp], dim=1).float().to(self.device)

                        batch_y_mark = batch_y_mark[:, self.input_chunk_length - self.decoder_length:self.input_chunk_length + self.output_chunk_length, :]
                    else:
                        #during prediction use transformer predictions as decoder input
                        dec_inp = torch.cat([imputes[:, -self.decoder_length:, :],
                                dec_inp], dim=1).float().to(self.device)
                        # batch_y_mark = batch_y_mark[:, self.input_chunk_length-dec_inp.size()[1]:self.input_chunk_length, :]
                        batch_y_mark = batch_y_mark[:, max(0, self.input_chunk_length-dec_inp.size()[1]+self.output_chunk_length):self.input_chunk_length+self.output_chunk_length, :]


                    _, outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                    outputs = outputs[:, :self.input_chunk_length, :]
                    batch_y = batch_y[:, :self.input_chunk_length, :].to(self.device)


                    if i == 0:
                        #uncomment if you want only to see imputed values
                        # imputes = outputs*seq_y_missing[:, :self.input_chunk_length, :] + batch_y*~seq_y_missing[:, :self.input_chunk_length, :]
                        imputes = outputs
                        truth = batch_y
                        missing_mask = seq_y_missing[:, :self.input_chunk_length, :]
                    else:
                        # imputation = outputs*seq_y_missing[:, :self.input_chunk_length, :] + batch_y*~seq_y_missing[:, :self.input_chunk_length, :]
                        imputation = outputs
                        imputes = torch.cat((imputes, imputation), dim=1)
                        truth = torch.cat((truth, batch_y), dim=1)

                        missing_mask = torch.cat((missing_mask, seq_y_missing[:, :self.input_chunk_length, :]), dim=1)


        imputes = imputes[:, :n_steps, :].cpu().view(n_steps, -1)
        targets = truth[:, :n_steps, :].cpu().view(n_steps, -1)
        missing_mask = missing_mask[:, :n_steps, :].cpu().view(n_steps, -1)



        return imputes.cpu().detach().numpy(), targets.cpu().detach().numpy(), missing_mask.cpu().detach().numpy()



