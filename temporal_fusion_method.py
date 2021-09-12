
import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters


def forecast_with_temporal_fusion(training_data, testing_data):
    print('traingin', training_data, testing_data)

    pl.seed_everything(42)
    trainer = pl.Trainer(
        gpus=0,
        gradient_clip_val=0.1,
    )

    shape = training_data.shape[0]
    print('column', training_data.columns)
    training_data['time_idx'] = range(0, shape)

    max_prediction_length = 6
    max_encoder_length = 24
    training_cutoff = training_data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        training_data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="open",
        group_ids=[],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["open"],        
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=7,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4,
    )

    validation = TimeSeriesDataSet.from_dataset(training, training_data, predict=True, stop_randomization=True)

    # create dataloaders for model
    batch_size = 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    res = trainer.tuner.lr_find(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    
    fig = res.plot(show=True, suggest=True)
    fig.show()

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")