# from lstm_method import forecast_with_lstm
import os
import sys
from lstm_method import forecast_with_lstm
from gru_method import forecast_with_gru
from transformer_multi_step import predict_with_transformer_multi
from transformer_single_step import predict_with_transformer_single

from prophet_method import forecast_with_prophet
# from temporal_fusion_method import forecast_with_temporal_fusion
# from activation_functions import activation_functions

os.chdir(sys.path[0])

from base_config import config
from utils import load_prediction_dataset

if __name__ == "__main__":
    training_data = load_prediction_dataset(
        ticker=config['ticker'],
        start_date=config['training_period_start_date'],
        end_date=config['training_period_end_date']
    )

    print('training', training_data)

    testing_data = load_prediction_dataset(
        ticker=config['ticker'],
        start_date=config['testing_period_start_date'],
        end_date=config['testing_period_end_date']
    )

    all_data = training_data.append(testing_data)

    # forecast_with_prophet(
    #     training_data=training_data,
    #     testing_data=testing_data
    # )

    # forecast_with_gru(
    #     training_dataset=training_data,
    #     testing_dataset=testing_data,
    #     all_data=all_data
    # )

    forecast_with_lstm(training_dataset=training_data, testing_dataset=testing_data, all_data=all_data)

    # predict_with_transformer_multi(
    #     training_data,
    #     testing_data
    # )

    # predict_with_transformer_single(
    #     training_data,
    #     testing_data
    # )