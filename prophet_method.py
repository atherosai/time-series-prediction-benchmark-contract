from utils import compare_metrics
from prophet import Prophet
from plot_forecasted_results import plot_forecasted_results
import matplotlib.pyplot as plt
from base_config import config
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import math, time
from utils import print_metrics

prophet_config = config['methods_hyper_config']['prophet']

def forecast_with_prophet(training_data, testing_data):

    for_prophet_training = training_data[['date', 'open']].rename(columns={
        'date': 'ds',
        'open': 'y'
    })

    for_prophet_testing = testing_data[['date', 'open']].rename(columns={
        'date': 'ds',
        'open': 'y'
    })

    m = Prophet(
        changepoint_prior_scale=prophet_config['changepoint_prior_scale'],
        changepoint_range=prophet_config['changepoint_range']
    )
    m.fit(for_prophet_training)

    period_length = datetime.datetime.strptime(config['testing_period_end_date'], '%Y-%m-%d') - datetime.datetime.strptime(
        config['testing_period_start_date'], '%Y-%m-%d')
    print('period length', period_length.days)
    future = m.make_future_dataframe(periods=period_length.days)

    forecast = m.predict(future)

    print('testing data', testing_data['open'].to_numpy())
    print('forecast: ', forecast['yhat'].to_numpy())

    # print(training_data['yhat'])
    # print(forecast['yhat'])

    training_data_reshape = testing_data['open'].to_numpy()
    forecast_data_reshape = forecast['yhat'][-252:].to_numpy()

    print('training size: ', training_data_reshape.size)
    print('forecast size: ', forecast_data_reshape.size)

    # print errors
    print_metrics(training_data_reshape, forecast_data_reshape)

    plot_forecasted_results(
        training_data=training_data,
        testing_data=testing_data,
        forecast=forecast,
        with_confidence_intervals=True    
    )

    return forecast