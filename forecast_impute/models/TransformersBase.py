import time

import numpy as np
import torch
from pypots.utils.metrics import cal_mse, cal_mae
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TransformersBase():
    """
    This is a base clase used to train the complex transformer model
    It is used as a interface which can be used for all tasks, imputation, forecasting and imputation+forecasting
    @Author MeelsL
    """

    def __init__(self, input_chunk_length: int, decoder_length: int, output_chunk_length: int, resample_rate:str, num_features=1,
                 batch_size: int = 32, n_epochs=100, d_model=64, dim_feedforward=512, num_layers=1, dropout=0.1, imputation=False, advanced_impute=False):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        self.input_chunk_length = input_chunk_length
        self.decoder_length = decoder_length
        self.output_chunk_length = output_chunk_length
        self.num_features = num_features
        self.batch_size = batch_size
        self.epochs = n_epochs
        self.imputation = imputation

        #whether to train the model on the advanced architecture or just the simple architecture
        self.advanced_impute = advanced_impute
        self.resample_rate = resample_rate

    def fit_loop(self, train_data, train_loader):
        """
        train the model for a certain number of epochs
        Training supports both tasks (forecasting only or imputation_forecast)
        :param train_data: training data
        :param train_loader: data_loader for the batches
        :return: trained model
        """

        print_freq = 10 if self.resample_rate=="h" else 1

        val_data, val_loader = train_data, train_loader

        lr = 0.005 #0.005

        forecast_optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        forecast_criterion = nn.MSELoss()
        # forecast_criterion = nn.L1Loss()
        forecast_scheduler = torch.optim.lr_scheduler.StepLR(forecast_optimizer, 1, gamma=0.95)

        if self.imputation and self.advanced_impute:
            imputation_criterion = cal_mse


        for epoch in tqdm(range(self.epochs)):
            train_loss = []
            iter_count = 0

            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_missing) in enumerate(train_loader):
                iter_count += 1

                forecast_optimizer.zero_grad()
                # if self.imputation and self.advanced_impute:
                #     imputation_optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                seq_y_missing = seq_y_missing.bool().to(self.device)

                batch_y_dec = batch_y.clone()
                batch_y_dec[seq_y_missing] = 0

                # decoder input
                dec_inp = torch.zeros_like(batch_y_dec[:, -self.output_chunk_length:, :]).float()
                dec_inp = torch.cat(
                    [batch_y_dec[:, self.input_chunk_length - self.decoder_length:self.input_chunk_length, :],
                     # -self.output_chunk_length
                     dec_inp], dim=1).float().to(self.device)

                batch_y_mark = batch_y_mark[:,
                               self.input_chunk_length - self.decoder_length:self.input_chunk_length + self.output_chunk_length,
                               :]

                # model forward
                outputs, imputes = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                # predict and compare
                outputs = outputs[:, -self.output_chunk_length:, :]
                batch_y_forecast = batch_y[:, -self.output_chunk_length:, :].to(self.device)

                forecast_loss = forecast_criterion(outputs, batch_y_forecast)

                if self.imputation and self.advanced_impute:
                    # advanced means we split the loss in two
                    imputation_loss = cal_mse(imputes, batch_y[:, :self.input_chunk_length, :],
                                              seq_y_missing[:, :self.input_chunk_length, :])
                    reconstruction_loss = cal_mse(imputes, batch_y[:, :self.input_chunk_length, :],
                                              ~seq_y_missing[:, :self.input_chunk_length, :])
                    imputation_loss = imputation_loss + reconstruction_loss




                #update weights only once
                if self.imputation and self.advanced_impute:
                    imputation_loss.backward(inputs= self.model.get_imputation_parameters(), retain_graph=False)


                forecast_loss.backward(inputs=self.model.get_forecast_parameters(), retain_graph=False)


                forecast_optimizer.step()


                if (i + 1) % print_freq == 0:
                    if self.imputation and self.advanced_impute:
                        print("\titers: {0}, epoch: {1} | forecast_loss: {2:.7f} | imputation_loss: {3:.7f}".format(i + 1, epoch + 1, forecast_loss.item(), imputation_loss.item()))
                    else:
                        print("\titers: {0}, epoch: {1} | forecast_loss: {2:.7f}".format(i + 1, epoch + 1, forecast_loss.item()))
                    iter_count = 0

            forecast_scheduler.step()

        return self.model

    def evaluate(self, val_loader, criterion):
        """
        model evaluations
        :param val_loader: validation data
        :param criterion: loss criterion
        :return: validation performance
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_missing) in enumerate(val_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                seq_y_missing = seq_y_missing.bool().to(self.device)

                batch_y_dec = batch_y.clone()
                batch_y_dec[seq_y_missing] = 0

                # decoder input
                dec_inp = torch.zeros_like(batch_y_dec[:, -self.output_chunk_length:, :]).float()
                dec_inp = torch.cat(
                    [batch_y_dec[:, self.input_chunk_length - self.decoder_length:self.input_chunk_length, :],
                     #:-self.output_chunk_length
                     dec_inp], dim=1).float().to(self.device)

                batch_y_mark = batch_y_mark[:,
                               self.input_chunk_length - self.decoder_length:self.input_chunk_length + self.output_chunk_length,
                               :]

                # predict and compare
                outputs, imputes = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = 0

                outputs = outputs[:, -self.output_chunk_length:, f_dim:]
                batch_y_forecast = batch_y[:, -self.output_chunk_length:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y_forecast.detach().cpu()

                forecast_loss = criterion(pred, true)

                if self.imputation and self.advanced_impute:
                    # Advanced means we split the two loses
                    imputation_loss = criterion(imputes.detach().cpu(), batch_y[:, :self.input_chunk_length, :])
                    loss = forecast_loss + imputation_loss
                else:
                    loss = forecast_loss

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def predict_loop(self, n_steps, input_loader:DataLoader):
        """
        Forecasts and imputes missing data at same time
        :param n_steps: number of steps to forecast
        :param input_loader: input loader
        :return: forecast and imputation
        """

        self.model.eval()
        with torch.no_grad():
            forecast = torch.zeros((1,1,1))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_y_missing) in enumerate(input_loader):

                #only predict for each timestep
                if i % self.output_chunk_length == 0:
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
                        dec_inp = torch.cat([forecast[:, -self.decoder_length:, :],
                                dec_inp], dim=1).float().to(self.device)
                        batch_y_mark = batch_y_mark[:, max(0, self.input_chunk_length-dec_inp.size()[1]+self.output_chunk_length):self.input_chunk_length+self.output_chunk_length, :]


                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs[:, -self.output_chunk_length:, :]
                    batch_y = batch_y[:, -self.output_chunk_length:, :].to(self.device)


                    if i ==0:
                        forecast = outputs
                        truth = batch_y
                        missing_mask = seq_y_missing[:, -self.output_chunk_length:, :]
                    else:
                        forecast = torch.cat((forecast, outputs), dim=1)
                        truth = torch.cat((truth, batch_y), dim=1)
                        missing_mask = torch.cat((missing_mask, seq_y_missing[:, -self.output_chunk_length:, :]), dim=1)

        forecast = forecast[:n_steps].cpu().view(n_steps, -1)
        targets = truth[:n_steps].cpu().view(n_steps, -1)
        missing_mask = missing_mask[:n_steps].cpu().view(n_steps, -1)


        return forecast.cpu().detach().numpy(), targets.cpu().detach().numpy(), missing_mask.cpu().detach().numpy()
