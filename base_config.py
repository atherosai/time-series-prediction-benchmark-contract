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
        'temporal_fusion': {
            'n_trials': 200,
            'max_epochs': 20,
            'gradient_clip_val': 0.1,
            'hidden_size': 128,
            'hidden_continuous_size': 8,
            'attention_head_size': 1,
            'learning_rate': 0.15135612484362077,
            'dropout': 0.1,
            # 'trainer_kwargs': dict(limit_train_batches=30),
            'reduce_on_plateau_patience': 4,
        }
    }
}