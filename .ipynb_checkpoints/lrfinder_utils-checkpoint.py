import torch 
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import Callback
from pytorch_forecasting.data import TimeSeriesDataSet

import pandas as pd

from tqdm import tqdm

import random

    def init_lr_finder(self, min_lr=1e-6):
        # Run learning rate finder
        lr_finder = self.trainer.tuner.lr_find(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=2.0,
            min_lr=min_lr,
            # mode='linear'
        )

        # Results can be found in
        lr_finder.results

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        self.tft.hparams.lr = new_lr
        self.tft.hparams.learning_rate = new_lr

        print('new_lr:', self.tft.hparams.lr)

        print(f"suggested learning rate: {lr_finder.suggestion()}")
        fig = lr_finder.plot(show=True, suggest=True)
        # fig.show()

        fig.tight_layout()
        fig.savefig(f'Dlr_find_[{self.predicted_years}]_[{self.batch_size}].png', dpi=300, format='png')
        
    def custom_finder(self, min_lr=1e-3):
        # Run learning rate finder
        self.trainer.tuner.lr_find = custom_lr_find
        # trainer.tuner.lr_find = custom_lr_find
        lr_finder = self.trainer.tuner.lr_find(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=1.0,
            min_lr=min_lr,
            mode='linear'
        )

        # Results can be found in
        print('lr_finder.results:', lr_finder.results)

        # # Plot with
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        self.tft.hparams.lr = new_lr
        self.tft.hparams.learning_rate = new_lr

        print('new_lr:', self.tft.hparams.lr)

        print(f"suggested learning rate: {lr_finder.suggestion()}")
        fig = lr_finder.plot(show=True, suggest=True)
        fig.show()

        fig.tight_layout()
        fig.savefig(f'Dcustom_find_[{self.predicted_years}]_[{self.batch_size}].png', dpi=300, format='png')
        
    def find_init_lr():
        # find optimal learning rate
        res = trainer.tuner.lr_find(
            self.tft,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
            max_lr=1.0,
            min_lr=1e-5,
        )

        print(f"suggested learning rate: {res.suggestion()}")
        fig = res.plot(show=True, suggest=True)
        fig.show()
        
        