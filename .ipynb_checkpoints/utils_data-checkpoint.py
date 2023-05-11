import torch 
from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import Callback

from pytorch_forecasting.data import TimeSeriesDataSet
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import MAPE, SMAPE
# from pytorch_forecasting.data import CombinedLoader

# from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

import random

# -----------------------------------------------------------------------------

# from torch.utils.tensorboard import SummaryWriter 

#-------------------------------------------------------------------------------------

class ReloadDataLoader(Callback):
    def __init__(self, train_dataset: TimeSeriesDataSet, batch_size: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.batch_size = batch_size
        
    def on_train_epoch_start(self, trainer, pl_module):
        # if trainer.current_epoch in self.milestones:
        #     print('ReloadDataLoader:', trainer.current_epoch)          
        pl_module.train_dataloader = self.train_dataset.to_dataloader(batch_size=self.batch_size, shuffle=True)
        print('DataLoader was reloaded...')
        
class ReloadDataSet(Callback):
    def __init__(self, data_train, dataset_train, dataloader_train, batch_size, YEARS_MAX_LENGTH=1, NSAMPLES=1):
        super().__init__()
        self.data_train = data_train
        # self.data_valid = data_valid
        self.dataset_train = dataset_train
        self.dataloader_train = dataloader_train
        self.batch_size = batch_size
        self.YEARS_MAX_LENGTH = YEARS_MAX_LENGTH
        self.NSAMPLES = NSAMPLES
        
    def on_train_epoch_start(self, trainer, pl_module):
        # if trainer.current_epoch in self.milestones:
        print('DataGenerator reloading... epoch:', trainer.current_epoch)  
        data_train, year_list = DataGenerator2(DATA=self.data_train, 
                                              YEARS_MAX_LENGTH=self.YEARS_MAX_LENGTH, 
                                              NSAMPLES=self.NSAMPLES)
        self.dataset_train = TimeSeriesDataSet.from_dataset(self.dataset_train, data_train)
        # pl_module.train_dataloader = self.dataset_train.to_dataloader(batch_size=self.batch_size, shuffle=True)
        self.dataloader_train = self.dataset_train.to_dataloader(batch_size=self.batch_size, shuffle=True)
        print('DataLoader was reloaded...')
        
class ReloadDataSet_12(Callback):
    def __init__(self, data_train, dataset_train, dataloader_train, batch_size, YEARS_MAX_LENGTH=1, NSAMPLES=1):
        super().__init__()
        self.data_train = data_train
        # self.data_valid = data_valid
        self.dataset_train = dataset_train
        self.dataloader_train = dataloader_train
        self.batch_size = batch_size
        self.YEARS_MAX_LENGTH = YEARS_MAX_LENGTH
        self.NSAMPLES = NSAMPLES
        
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch <= 2:
            print('DataGenerator_1 reloading... epoch:', trainer.current_epoch)  
            data_train, year_list = DataGenerator2(DATA=self.data_train, 
                                                  YEARS_MAX_LENGTH=self.YEARS_MAX_LENGTH, 
                                                  NSAMPLES=self.NSAMPLES)
            self.dataset_train = TimeSeriesDataSet.from_dataset(self.dataset_train, data_train)
            
            self.dataloader_train = self.dataset_train.to_dataloader(train=True, 
                                                                batch_size=self.batch_size, 
                                                                shuffle=True)
            pl_module.train_dataloader = lambda: self.dataloader_train
            # self.dataloader_train = self.dataset_train.to_dataloader(train=True, 
            #                                                     batch_size=self.batch_size, 
            #                                                     shuffle=True)
            # trainer.train_dataloader = self.dataloader_train
            print(f'DataLoader_1 was reloaded... {len(data_train)}, {len(year_list)}, {len(trainer.train_dataloader)},')#{len(pl_module.train_dataloader)}')
        elif trainer.current_epoch > 2:
            print('DataGenerator_2 reloading... epoch:', trainer.current_epoch)  
            data_train, year_list = DataGenerator2(DATA=self.data_train, 
                                                  YEARS_MAX_LENGTH=self.YEARS_MAX_LENGTH, 
                                                  NSAMPLES=self.NSAMPLES)
            self.dataset_train = TimeSeriesDataSet.from_dataset(self.dataset_train, data_train)
            
            self.dataloader_train = self.dataset_train.to_dataloader(train=True, 
                                                                batch_size=self.batch_size, 
                                                                shuffle=True)
            pl_module.train_dataloader = lambda: self.dataloader_train
            # trainer.train_dataloader = lambda: self.dataloader_train
            # trainer.train_dataloader = self.dataset_train.to_dataloader(train=True, 
            #                                                     batch_size=self.batch_size, 
            #                                                     shuffle=True)
            print(f'DataLoader_2 was reloaded... {len(data_train)}, {len(year_list)}, {len(trainer.train_dataloader)},')#{len(pl_module.train_dataloader)}')

################################################################################################# 

def DataGenerator(DATA, YEARS_MAX_LENGTH, NSAMPLES):
    years_list = list(DATA['year'].astype(int).unique())
    print(f'Augmentation for years list: {years_list} by NSAMPLES={NSAMPLES} and YEARS_MAX_LENGTH={YEARS_MAX_LENGTH}')

    data_samples = pd.DataFrame()
    years_samples = []
    for ii in tqdm(range(NSAMPLES)):
        for county in DATA["county"].unique():
            # generate random number of trainig years
            num_years = random.randint(1, YEARS_MAX_LENGTH)
            # get list of training years 
            years = random.sample(years_list, num_years)
            years_samples.append(years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, DATA.loc[ (DATA['year'].astype(int) == iyear) & \
                                                         (DATA['county'] == county)] ], axis=0)
            # reindex the concatenated dataframe with a new index
            new_index = pd.RangeIndex(start=0, stop=len(df_concat_year)+0, step=1)
            df_concat_year.index = new_index
            # add a new column with integer values equal to the index
            df_concat_year["time_idx"] = df_concat_year.index.astype(int)
            df_concat_year["sample"] = str(ii)
            data_samples = pd.concat([data_samples, df_concat_year], axis=0)
        # reindex the concatenated dataframe with a new index
    new_index = pd.RangeIndex(start=0, stop=len(data_samples)+0, step=1)
    data_samples.index = new_index

    return data_samples, years_samples

################################################################################################# 

def DataGenerator2(DATA, YEARS_MAX_LENGTH, NSAMPLES):
    years_list = list(DATA['year'].astype(int).unique())
    print(f'DataGenerator2: Augmentation for years list: {years_list} by NSAMPLES={NSAMPLES}')

    data_samples = pd.DataFrame()
    years_samples = []
    for ii in tqdm(range(NSAMPLES)):
        for county in DATA["county"].unique():
            # generate random number of trainig years
            # num_years = len(years_list)  # random.randint(1, YEARS_MAX_LENGTH)
            # get list of training years 
            # years = random.sample(years_list, num_years)
            years = years_list
            random.shuffle(years)
            # print('DataGenerator2:', years)
            # fn
            years_samples.append(years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, DATA.loc[ (DATA['year'].astype(int) == iyear) & \
                                                                       (DATA['county'] == county)] ], 
                                           axis=0)
            # reindex the concatenated dataframe with a new index
            new_index = pd.RangeIndex(start=0, stop=len(df_concat_year)+0, step=1)
            df_concat_year.index = new_index
            # add a new column with integer values equal to the index
            df_concat_year["time_idx"] = df_concat_year.index.astype(int)
            df_concat_year["sample"] = str(ii)
            data_samples = pd.concat([data_samples, df_concat_year], axis=0)
        # reindex the concatenated dataframe with a new index
    new_index = pd.RangeIndex(start=0, stop=len(data_samples)+0, step=1)
    data_samples.index = new_index

    return data_samples, years_samples

################################################################################################# 

def DataGenerator3(DATA, YEARS_MAX_LENGTH, NSAMPLES):
    years_list = list(DATA['year'].astype(int).unique())
    print(f'DataGenerator2: Augmentation for years list: {years_list} by NSAMPLES={NSAMPLES}')

    data_samples = pd.DataFrame()
    years_samples = []
    for ii in tqdm(range(NSAMPLES)):
        for county in DATA["county"].unique():
            # generate random number of trainig years
            # num_years = len(years_list)  # random.randint(1, YEARS_MAX_LENGTH)
            # get list of training years 
            # years = random.sample(years_list, num_years)
            years = years_list
            # random.shuffle(years)
            # print('DataGenerator2:', years)
            # fn
            years_samples.append(years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, DATA.loc[ (DATA['year'].astype(int) == iyear) & \
                                                                       (DATA['county'] == county)] ], 
                                           axis=0)
            # reindex the concatenated dataframe with a new index
            new_index = pd.RangeIndex(start=0, stop=len(df_concat_year)+0, step=1)
            df_concat_year.index = new_index
            # add a new column with integer values equal to the index
            df_concat_year["time_idx"] = df_concat_year.index.astype(int)
            df_concat_year["sample"] = str(ii)
            data_samples = pd.concat([data_samples, df_concat_year], axis=0)
        # reindex the concatenated dataframe with a new index
    new_index = pd.RangeIndex(start=0, stop=len(data_samples)+0, step=1)
    data_samples.index = new_index

    return data_samples, years_samples


