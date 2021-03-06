import os

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import MyDataset
from model.conv_lstm import ConvLSTM
from utils.earlystoping import EarlyStopping
from utils.getdata import get_data


class EXP:
    def __init__(self, his_len=10, pre_len=10, lr=0.001, batch_size=8, epochs=10, patience=3, verbose=True):
        self.his_len = his_len
        self.pre_len = pre_len

        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.lr = lr

        if not os.path.exists('./checkpoint/'):
            os.makedirs('./checkpoint/')

        self.modelpath = './checkpoint/convlstm_model.pkl'

        self._get_data()
        self._get_model()

    def _get_data(self):
        train, valid, test = get_data()

        trainset = MyDataset(train, his_len=self.his_len, pre_len=self.pre_len)
        validset = MyDataset(valid, his_len=self.his_len, pre_len=self.pre_len)
        testset = MyDataset(test, his_len=self.his_len, pre_len=self.pre_len)

        self.trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        self.validloader = DataLoader(validset, batch_size=self.batch_size, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)

    def _get_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvLSTM().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.75 ** ((epoch - 1) // 2))
        self.early_stopping = EarlyStopping(patience=self.patience, verbose=self.verbose, path=self.modelpath)
        self.criterion = nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)  # BatchSize * HisLen * P * Weight * Height
        batch_y = batch_y.float().to(self.device)  # BatchSize * HisLen * P * Weight * Height

        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)
        return outputs, loss

    def train(self):
        for e in range(self.epochs):
            self.model.train()
            train_loss = []
            for (batch_x, batch_y) in tqdm(self.trainloader):
                self.optimizer.zero_grad()
                pred, loss = self._process_one_batch(batch_x, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_loss = []
                for (batch_x, batch_y) in tqdm(self.validloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    valid_loss.append(loss.item())

                test_loss = []
                for (batch_x, batch_y) in tqdm(self.testloader):
                    pred, loss = self._process_one_batch(batch_x, batch_y)
                    test_loss.append(loss.item())

            train_loss, valid_loss, test_loss = np.average(train_loss), np.average(valid_loss), np.average(test_loss)
            print("Epoch: {0}, | Train Loss: {1:.4f} Vali Loss: {2:.4f} Test Loss: {3:.4f}".format(e + 1, train_loss,
                                                                                                   valid_loss,
                                                                                                   test_loss))

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break
            self.scheduler.step()
        self.model.load_state_dict(torch.load(self.modelpath))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            trues, preds = [], []
            for (batch_x, batch_y) in tqdm(self.testloader):
                pred, loss = self._process_one_batch(batch_x, batch_y)
                preds.extend(pred.detach().cpu().numpy())
                trues.extend(batch_y.detach().cpu().numpy())

        trues, preds = np.array(trues), np.array(preds)
        mae = np.mean(np.abs(preds - trues))
        mse = np.mean((preds - trues) ** 2)
        print('Test: MSE:{0:.4f}, MAE:{1:.6f}'.format(mse, mae))
