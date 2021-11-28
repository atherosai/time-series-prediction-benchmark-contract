# SPY OXY TSLA AMZN
config = {
    'ticker': 'AMZN',
    'training_period_start_date': '2010-01-01',
    'training_period_end_date': '2016-12-31',
    'testing_period_start_date': '2017-01-01',
    'testing_period_end_date': '2017-12-31',
    'methods_hyper_config': {
        'prophet': {
            "changepoint_prior_scale": 0.3,
            'changepoint_range': 0.9
        },
        'gru': {
            'reduction': 'mean',
            "lookback": 20,
            "batch_size": 20,
            "input_dim": 1,
            "hidden_dim": 32,
            "num_layers": 2,
            "output_dim": 1,
            "num_epochs": 100,
            'lr': 0.01
        },
        'lstm': {
            'reduction': 'mean',
            "lookback": 20,
            "batch_size": 20,
            "input_dim": 1,
            "hidden_dim": 32,
            "num_layers": 2,
            "output_dim": 1,
            "num_epochs": 100,
            'lr': 0.01
        },
        'transformer': {
            'calculate_loss_over_all_values': True,
            'input_window': 150,
            'output_window': 10,
            'learning_rate': 0.001,
            'number_of_epochs': 500,
            'batch_size': 10,
            'feature_size': 250,
            'num_layers': 1,
            'dropout': 0.1,
            'nhead': 10,
            'eval_batch_size': 1000,
        },
    }
}

# transformers, lr 0.1 - SPY, feature size 500, nhead 20

# 150 epoch
# MAE:  0.9629993286132813
# MAPE:  0.0037760472218508908
# RMSE:  1.8780564973463296


# transformers, lr 0.001 - SPY, feature size 500, nhead 20

# 30 epoch
# MAE:  1.0448790893554687
# MAPE:  0.004083262556330397
# RMSE:  1.9094796014438613


# transformers, lr 0.01 - SPY, feature size 250, nhead 10

# 10 epoch
# MAE:  1.4769801635742188
# MAPE:  0.005844707243743297
# RMSE:  2.560467234267184

# 20 epoch
# MAE:  1.0551901245117188
# MAPE:  0.0041320751185674135
# RMSE:  2.046450169797463


# transformers, lr 0.001 - SPY, feature size 250, nhead 10

# 80 epoch
# MAE:  0.6938192749023437
# MAPE:  0.0027512668177530424
# RMSE:  1.3838778789384778

# transformers, lr 0.0005 - SPY, feature size 250, nhead 10

# 40 epoch
# MAE:  1.049423095703125
# MAPE:  0.004157169789191488
# RMSE:  1.962772093814715

# 60 epoch
# MAE:  1.1963215942382812
# MAPE:  0.0047053497737298495
# RMSE:  2.078176938420506
