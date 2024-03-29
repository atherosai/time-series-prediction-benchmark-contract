import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from base_config import config
import matplotlib.pyplot as plt
import seaborn as sns
import math, time
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from utils import print_metrics

lstm_config = config['methods_hyper_config']['lstm']

lookback = lstm_config['lookback']

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

model = LSTM(
    input_dim=lstm_config['input_dim'],
    hidden_dim=lstm_config['hidden_dim'],
    output_dim=lstm_config['output_dim'],
    num_layers=lstm_config['num_layers']
)

criterion = torch.nn.MSELoss(reduction=lstm_config['reduction'])
optimiser = torch.optim.Adam(model.parameters(), lr=lstm_config['lr'])


def split_data(price, lookback, test_set_size):
    data_raw = price.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

scaler = MinMaxScaler(feature_range=(-1, 1))

def process_to_tensor_procedure(data_set, test_set_size):

    price = data_set[['open']]

    price['open'] = scaler.fit_transform(price['open'].values.reshape(-1,1))

    x_train, y_train, x_test, y_test = split_data(price, lookback, test_set_size)

    return [
       torch.from_numpy(x_train).type(torch.Tensor),
       torch.from_numpy(y_train).type(torch.Tensor),
       torch.from_numpy(x_test).type(torch.Tensor),
       torch.from_numpy(y_test).type(torch.Tensor),
       price
    ]

def forecast_with_lstm(training_dataset, testing_dataset, all_data):
    num_epochs = 100
    
    hist = np.zeros(num_epochs)
    start_time = time.time()

    [x_train, y_train, x_test, y_test, price] = process_to_tensor_procedure(all_data, len(testing_dataset['open']))
    
    for t in range(num_epochs):
        
        y_train_pred = model(x_train)

        loss = criterion(y_train_pred, y_train)
        
        print('Epocha', t, loss)

        hist[t] = loss.item()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    
    y_test_pred = model(x_test)

    training_time = time.time() - start_time

    predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
    original = pd.DataFrame(scaler.inverse_transform(y_train.detach().numpy()))

    print('trainignti', training_time)

    sns.set_style("darkgrid")    

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    plt.subplot(1, 2, 1)
    ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Predikce na trénovacích datech (LSTM)", color='tomato')
    ax.set_title(f"{config['ticker']}", size = 14, fontweight='bold')
    ax.set_xlabel("Datum", size = 14)
    ax.set_ylabel("Cena při otevření burzy $", size = 14)
    ax.set_xticklabels('', size=10)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epocha", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Loss při tréningu LSTM", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

    plt.show()

    # make predictions
    y_test_pred = model(x_test)

    lstm = []

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # print errors
    print_metrics(y_test[:,0], y_test_pred[:,0])

    # lstm.append(trainScore)
    # lstm.append(testScore)
    lstm.append(training_time)

    print(y_train_pred)

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price['open'].values.reshape(-1,1))

    print('trainPredictPlot: ', trainPredictPlot)
    print('testPredictPlot: ', testPredictPlot)
    print('original: ', original)

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    plt.title(f" {config['ticker']} LSTM predikce")

    plot_args = (
        all_data['date'], result[0], 'blue',
        all_data['date'],  result[1], 'red',
        all_data['date'], result[2], 'violet'
    )

    plt.plot(
        *plot_args
    )

    plt.legend(
        [
            'Trénovací datový soubor',
            'Testovací datový soubor',
            'Predikce',
            # 'Horní hranice predikce',
            # 'Dolní hranice predikce'
        ]
    )

    plt.xlabel(f" Datum")
    plt.ylabel(f" {config['ticker']} cena při otevření burzy $")

    plt.show()

