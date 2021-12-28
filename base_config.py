# SPY OXY TSLA AMZN
config = {
    'ticker': 'SPY',
    'training_period_start_date': '2012-01-01',
    'training_period_end_date': '2016-12-31',
    'testing_period_start_date': '2017-01-01',
    'testing_period_end_date': '2017-12-31',
    'methods_hyper_config': {
        'prophet': {
            "changepoint_prior_scale": 0.3,
            'changepoint_range': 0.9
        },
        'gru': {
            'lr': 0.01,
            "num_epochs": 100,
            "input_dim": 1,
            "hidden_dim": 32,
            "output_dim": 1,
            "num_layers": 2,
            "batch_size": 20,
            "lookback": 20,
            'reduction': 'mean',
        },
        'lstm': {
            'lr': 0.01,
            "num_epochs": 100,
            "input_dim": 1,
            "hidden_dim": 32,
            "output_dim": 1,
            "num_layers": 2,
            "batch_size": 20,
            "lookback": 20,
            'reduction': 'mean',
        },
        'transformer': {
            'lr': 0.1,
            'num_epochs': 100,
            'input_window': 1,
            'output_window': 1,
            'hidden_dim': 32,
            'num_layers': 1,
            'batch_size': 1,
            'dropout': 0.1,
            'nhead': 8,
            'eval_batch_size': 1000,
            'calculate_loss_over_all_values': True,
        },
    }
}

# y_true.size:  298
# y_pred.size:  298

# SPY
# transformers, lr 0.01 - SPY, hidden size 128, nhead 8

# MAE:  1.028805419921875
# MAPE:  0.004151119179610707
# RMSE:  1.3494785401665705
# Epoch:  60

# transformers, lr 0.05 - SPY, hidden size 128, nhead 8

# MAE:  0.93498779296875
# MAPE:  0.0037750419300992036
# RMSE:  1.3396207970761123
# Epoch:  170




# OXY
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  0.8890245208740234
# MAPE:  0.012743659299718204
# RMSE:  2.283644642099354
# Epoch:  10




# TSLA
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  8.283213073730469
# MAPE:  0.026254763891988938
# RMSE:  19.118175630426077
# Epoch:  20




# AMZN
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  5.963873046875
# MAPE:  0.005755070370978157
# RMSE:  17.177625540641323
# Epoch:  10

