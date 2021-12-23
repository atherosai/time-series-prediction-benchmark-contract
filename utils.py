import re
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import math, time

yf.pdr_override()

def to_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def pd_to_underscore(pd_to_rename, columns=[]):
    rename_map = {}
    
    if (len(columns) == 0):
        columns = list(pd_to_rename)

    for column in columns:
        rename_map[column] = to_underscore(column)
    
    pd_renamed = pd_to_rename.rename(index=str, columns=rename_map)

    return pd_renamed

def load_prediction_dataset(
        ticker,
        start_date,
        end_date
    ):

    fetched_data = pdr.get_data_yahoo(
        ticker,
        start=start_date,
        end=end_date
    )

    fetched_data['date'] = fetched_data.index

    return pd_to_underscore(fetched_data)

def print_metrics(y_true, y_pred):

    print('y_true.size: ', y_true.size)
    print('y_pred.size: ', y_pred.size)

    y_true_trimmed = y_true[-250:]
    y_pred_trimmed = y_pred[-250:]

    print(y_true_trimmed.size)
    print(y_pred_trimmed.size)

    # print(y_true_trimmed)
    # print(y_pred_trimmed)
    
    print('MAE: ', mean_absolute_error(y_true_trimmed, y_pred_trimmed))
    print('MAPE: ', mean_absolute_percentage_error(y_true_trimmed, y_pred_trimmed))
    print('RMSE: ', math.sqrt(mean_squared_error(y_true_trimmed, y_pred_trimmed)))

def compare_metrics(y_true, y_pred):
    mean_absolute_error_result = mean_absolute_error(y_true, y_pred)
    mean_squared_error_result = mean_squared_error(y_true, y_pred)

    return {
        "mean_absolute_error_result": mean_absolute_error_result,
        "mean_squared_error_result": mean_squared_error_result
    }
