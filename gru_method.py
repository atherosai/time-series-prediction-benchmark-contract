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

gru_config = config['methods_hyper_config']['gru']

lookback = gru_config['lookback']

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

model = GRU(
    input_dim=gru_config['input_dim'],
    hidden_dim=gru_config['hidden_dim'],
    output_dim=gru_config['output_dim'],
    num_layers=gru_config['num_layers']
)

criterion = torch.nn.MSELoss(reduction=gru_config['reduction'])
optimiser = torch.optim.Adam(model.parameters(), lr=gru_config['lr'])


def split_data(price, lookback):
    data_raw = price.to_numpy() # convert to numpy array
    data = []
    
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

scaler = MinMaxScaler(feature_range=(-1, 1))

def process_to_tensor_procedure(data_set):

    price = data_set[['open']]

    price['open'] = scaler.fit_transform(price['open'].values.reshape(-1,1))

    x_train, y_train, x_test, y_test = split_data(price, lookback)

    return [
       torch.from_numpy(x_train).type(torch.Tensor),
       torch.from_numpy(y_train).type(torch.Tensor),
       torch.from_numpy(x_test).type(torch.Tensor),
       torch.from_numpy(y_test).type(torch.Tensor),
       price
    ]

def forecast_with_gru(training_dataset, testing_dataset, all_data):
    num_epochs = 100
    
    hist = np.zeros(num_epochs)
    start_time = time.time()

    [x_train, y_train, x_test, y_test, price] = process_to_tensor_procedure(all_data)
    
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
    ax = sns.lineplot(x = predict.index, y = predict[0], label="Predikce na trénovacích datech (GRU)", color='tomato')
    ax.set_title(f"{config['ticker']}", size = 14, fontweight='bold')
    ax.set_xlabel("Datum", size = 14)
    ax.set_ylabel("Cena při otevření burzy $", size = 14)
    ax.set_xticklabels('', size=10)

    plt.subplot(1, 2, 2)
    ax = sns.lineplot(data=hist, color='royalblue')
    ax.set_xlabel("Epocha", size = 14)
    ax.set_ylabel("Loss", size = 14)
    ax.set_title("Loss při tréningu GRU", size = 14, fontweight='bold')
    fig.set_figheight(6)
    fig.set_figwidth(16)

    plt.show()

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # print errors
    print_metrics(y_test[:,0], y_test_pred[:,0])

    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(price)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

    # shift test predictions for plotting
    testPredictPlot = np.empty_like(price)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(y_train_pred)+lookback-1:len(price)-1, :] = y_test_pred

    original = scaler.inverse_transform(price['open'].values.reshape(-1,1))

    predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
    predictions = np.append(predictions, original, axis=1)
    result = pd.DataFrame(predictions)

    plt.title(f" {config['ticker']} GRU predikce")

    plot_args = (
        result.index, result[0], 'blue',
        result.index,  result[1], 'red',
        result.index, result[2], 'violet'
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

    # fig = go.Figure()


    # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
    #                     mode='lines',
    #                     name='Predikce pro tréninková data')))
    # fig.add_trace(go.Scatter(x=result.index, y=result[1],
    #                     mode='lines',
    #                     name='Predikce pro testovací data'))
    # fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
    #                     mode='lines',
    #                     name='Reálná hodnota')))
    # fig.update_layout(
    #     xaxis=dict(
    #         showline=True,
    #         showgrid=True,
    #         showticklabels=False,
    #         linecolor='white',
    #         linewidth=2
    #     ),
    #     yaxis=dict(
    #         title_text='Cena při otevření burzy $',
    #         titlefont=dict(
    #             family='Rockwell',
    #             size=12,
    #             color='white',
    #         ),
    #         showline=True,
    #         showgrid=True,
    #         showticklabels=True,
    #         linecolor='white',
    #         linewidth=2,
    #         ticks='outside',
    #         tickfont=dict(
    #             family='Rockwell',
    #             size=12,
    #             color='white',
    #         ),
    #     ),
    #     showlegend=True,
    # )

    # annotations = []
    # annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
    #                             xanchor='left', yanchor='bottom',
    #                             text='Výsledky (GRU)',
    #                             font=dict(family='Rockwell',
    #                                         size=26,
    #                                         color='white'),
    #                             showarrow=False))
    # fig.update_layout(annotations=annotations)

    fig.show()


    # y_train = scaler_train.inverse_transform(y_train.detach().numpy())
    # y_train_pred = scaler_train.inverse_transform(y_train_pred.detach().numpy())
    # x_test = scaler_test.inverse_transform(x_test.detach().numpy())
    # y_test = scaler_test.inverse_transform(y_test.detach().numpy())
    # y_test_pred = scaler_test.inverse_transform(y_test_pred.detach().numpy())

    # train_predict_plot = np.empty_like(y_train_pred)
    # train_predict_plot[:, :] = y_train_pred
    # # train_predict_plot[lookback: len(y_train_pred), :-lookback] = y_train_pred

    # test_predict_plot = np.empty_like(y_test_pred)
    # test_predict_plot[:, :] = y_test_pred
    # test_predict_plot[len(y_train_pred) + lookback - 1: testing_dataset['open'].shape[0]-1, :] = y_test_pred

    # print('shapes', train_predict_plot.shape, test_predict_plot.shape)

    # predictions = np.append(
    #     train_predict_plot,
    #     test_predict_plot,
    #     axis=0)

    # print('predictions', predictions.shape)
    

    # result = pd.DataFrame(predictions)

    # print('result', result)
