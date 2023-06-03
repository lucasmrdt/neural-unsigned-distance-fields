import wandb
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from datasets import Shape2D
from model import NDF


def train(data_folder='data', **kwargs):
    train_dataloader = th.utils.data.DataLoader(
        Shape2D(folder=data_folder, mode='train'), batch_size=10, shuffle=True)
    val_dataloader = th.utils.data.DataLoader(
        Shape2D(folder=data_folder, mode='val'), batch_size=4, shuffle=False)

    model = NDF(**kwargs)

    wandb_logger = WandbLogger(project='ndf')
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{wandb.run.name}', save_top_k=5, monitor='val_loss')
    ei_early_stop = EarlyStopping(monitor='val_loss', patience=20)

    trainer = pl.Trainer(accelerator='gpu', devices=[1], max_epochs=200,
                         logger=wandb_logger, callbacks=[
        checkpoint_callback, ei_early_stop])

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    args = [
        {'loss': 'l2', 'lr': 1e-4, 'clamp': True},
        {'loss': 'l2', 'lr': 1e-4, 'clamp': False},
        {'loss': 'l1', 'lr': 1e-4, 'clamp': True},
        {'loss': 'l1', 'lr': 1e-4, 'clamp': False},
    ]
    data_folders = [
        'data-3',
        'data-4',
        'data-5',
        'data-6',
    ]
    for data_folder in data_folders:
        for arg in args:
            wandb.init(project='ndf',
                       config={'data_folder': data_folder, **arg})
            train(data_folder=data_folder, **arg)
            wandb.finish(quiet=True)
