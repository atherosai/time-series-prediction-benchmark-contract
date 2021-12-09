# SPY OXY TSLA AMZN
config = {
    'ticker': 'TSLA',
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
            'lr': 0.01,
            'num_epochs': 200,
            'input_window': 190,
            'hidden_dim': 128,
            'num_layers': 1,
            'batch_size': 10,
            'output_window': 1,
            # hidden dimension
            'dropout': 0.1,
            'nhead': 16,
            'eval_batch_size': 1000,
            'calculate_loss_over_all_values': True,
        },
    }
}

# SPY
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  0.3626947021484375
# MAPE:  0.0014059381541221593
# RMSE:  0.890154642640292
# Epoch:  100

# OXY
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  0.7391075134277344
# MAPE:  0.010655233428303584
# RMSE:  1.782664781056776
# Epoch:  20

# TSLA
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  1.2078526916503907
# MAPE:  0.0193124832795367
# RMSE:  3.0113368285156796
# Epoch:  10


# AMZN
# transformers, lr 0.01 - SPY, feature size 128, nhead 16

# MAE:  9.1003515625
# MAPE:  0.008156788804576723
# RMSE:  21.688824980957495
# Epoch:  100

