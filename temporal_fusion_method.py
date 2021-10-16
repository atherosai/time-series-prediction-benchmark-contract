
import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, MASE, RMSE, SMAPE, PoissonLoss, QuantileLoss

import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from base_config import config

transformer_config = config['methods_hyper_config']['temporal_fusion']

def forecast_with_temporal_fusion(training_dataset, testing_dataset, all_data):
    print('training_dataset: ', training_dataset);

    shape = training_dataset.shape[0]
    training_dataset['time_idx'] = range(0, shape)
    training_dataset['group'] = 'group'

    max_prediction_length = 1000
    max_encoder_length = 24
    training_cutoff = training_dataset["time_idx"].max() - max_prediction_length

    print('training_dataset', training_dataset)
    print('training_cutoff', training_cutoff)

    training = TimeSeriesDataSet(
        # training_dataset,
        training_dataset[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="open",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        time_varying_known_reals=["time_idx", "volume"],
    )

    validation = TimeSeriesDataSet.from_dataset(training, training_dataset, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=24)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=24)

    # configure network and trainer
    pl.seed_everything(42)
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=transformer_config['max_epochs'],
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        # fast_dev_run=True,
        gradient_clip_val=transformer_config['gradient_clip_val'],
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=transformer_config['learning_rate'],
        hidden_size=transformer_config['hidden_size'],  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=transformer_config['attention_head_size'],
        dropout=transformer_config['dropout'],  # between 0.1 and 0.3 are good values
        hidden_continuous_size=transformer_config['hidden_continuous_size'],  # set to <= hidden_size
        output_size=7,  # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience=transformer_config['reduce_on_plateau_patience'],
    )

    # # find optimal learning rate
    # res = trainer.tuner.lr_find(
    #     tft,
    #     train_dataloader=train_dataloader,
    #     val_dataloaders=val_dataloader,
    #     max_lr=10.0,
    #     min_lr=1e-6,
    # )

    # print(f"suggested learning rate: {res.suggestion()}")
    # fig = res.plot(show=True, suggest=True)
    # fig.show()

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # find_hyperparameters(train_dataloader, val_dataloader)

    # load the best model according to the validation loss (given that
    # we use early stopping, this is not necessarily the last epoch)
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
    raw_predictions, x = best_tft.predict(val_dataloader, return_x=True)

    fig = plt.figure()

    forecast = raw_predictions

    # plot_forecasted_results(training_dataset, testing_dataset, x, raw_predictions)

    actuals = torch.cat([y for x, (y, weight) in iter(val_dataloader)])
    predictions, x = best_tft.predict(val_dataloader, return_x=True)
    predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
    best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)

    print_metrics(predictions, actuals)

    plt.show()

def find_hyperparameters(train_dataloader, val_dataloader):
    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)

def print_metrics(predictions, actuals):
    # MAE, MAPE, MASE, RMSE, SMAPE
    mae = MAE(reduction="none")(predictions, actuals).mean(1)
    mape = MAPE(reduction="none")(predictions, actuals).mean(1)
    rmse = RMSE(reduction="none")(predictions, actuals).mean(1)
    # smape = SMAPE(reduction="none")(predictions, actuals).mean(1)

    print('MAE: ', mae)
    print('MAPE: ', mape)
    print('RMSE: ', rmse)
    # print('SMAPE: ', smape)

# def plot_forecasted_results(
#         training_dataset,
#         testing_dataset,
#         x,
#         raw_predictions,
#     ):

#     # all true values for y of the first sample in batch
#     encoder_targets = to_list(x["encoder_target"])
#     decoder_targets = to_list(x["decoder_target"])

#     plt.title(f" Predikce")

#     # plot_args = (
#     #     training_dataset['date'], training_dataset['open'], 'blue',
#     #     testing_dataset['date'], testing_dataset['open'], 'red',
#     #     testing_dataset['date'], [raw_predictions.numpy()], 'violet'
#     # )

#     # print(plot_args);

#     # plt.plot(
#     #     *plot_args
#     # )

#     # plt.legend(
#     #     ['Trénovací datový soubor', 'Testovací datový soubor', 'Predikce']
#     # )

#     # plt.xlabel(f" Datum")
#     # plt.ylabel(f" {config['ticker']} cena při otevření burzy $")

#     # plt.show()

# # def process_to_tensor_procedure(data_set):
# #     price = data_set[['open']]

# #     price['open'] = scaler.fit_transform(price['open'].values.reshape(-1,1))

# #     x_train, y_train, x_test, y_test = split_data(price, lookback)

# #     return [
# #        torch.from_numpy(x_train).type(torch.Tensor),
# #        torch.from_numpy(y_train).type(torch.Tensor),
# #        torch.from_numpy(x_test).type(torch.Tensor),
# #        torch.from_numpy(y_test).type(torch.Tensor),
# #        price
# #     ]
