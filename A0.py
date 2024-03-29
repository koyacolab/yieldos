import os

# home_dir = '/content/gdrive/My Drive/AChina' 
# home_dir = '/hy-tmp'
# os.chdir(home_dir)
# pwd

# pip install tqdm 

from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

import os
import warnings
import sys

import fire

# warnings.filterwarnings("ignore")  # avoid printing out absolute paths

# os.chdir("../../..")

#pip install scipy
#pip install torch pytorch-lightning pytorch_forecasting

import copy
from pathlib import Path
import warnings

import time

import random

import numpy as np
import pandas as pd
import lightning.pytorch  as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import MAPE, SMAPE, PoissonLoss, QuantileLoss, RMSE, MAE, MASE
from matplotlib import pyplot as plt
from lightning.pytorch.utilities.model_summary import summarize
from pytorch_forecasting import TemporalFusionTransformer
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from matplotlib import pyplot as plt

from pytorch_forecasting.metrics import MultiHorizonMetric  #, CompositeMetric

from multiprocessing import Pool, freeze_support

from multiprocessing import Pool, freeze_support

from lightning.pytorch.callbacks import LearningRateFinder

from lightning.pytorch.callbacks import GradientAccumulationScheduler

from utils import FineTuneLearningRateFinder_CyclicLR, FineTuneLearningRateFinder_LinearLR, FineTuneLearningRateFinder_CustomLR
from utils import FineTuneLearningRateFinder_CyclicLR2, Reseter, FineTuneLearningRateFinder_CustomLR2
from utils import FineTuneLearningRateFinder_MultiStepLR
from utils_data import ReloadDataLoader, ReloadDataSet, ReloadDataSet_12
from utils_data import DataGenerator, DataGenerator2, DataGenerator3
from utils import ActualVsPredictedCallback
    
from pytorch_forecasting.metrics import MultiHorizonMetric

# from LRCustom import custom_lr_find

class Myloss(MultiHorizonMetric):
    """
    sqrtRMSE + MAE

    Defined as ``sqrt{(y_pred - target)**2} + (y_pred - target).abs()``
    """
        
    def loss(self, y_pred, target):
        loss = ( torch.pow(self.to_prediction(y_pred) - target, 2) / \
                torch.pow(self.to_prediction(y_pred) - target.mean(), 2) )
        return loss
    
# MOD_BINS = 512
# FAM_BINS = 256

# MOD_BINS = 64
# FAM_BINS = 32

MOD_BINS = 32
FAM_BINS = 16
CROP = 'corn'

class ModelBase:
    
    def __init__(self, 
                 home_dir = '/hy-tmp',
                 # datasetfile = f'data/ALIM{MOD_BINS}F{FAM_BINS}DATASET_rice.csv',
                 # datasetfile = f'data/AdB_M{MOD_BINS}_F{FAM_BINS}DATASET_rice.csv',    
                 # datasetfile = f'data/ANdB_M{MOD_BINS}_F{FAM_BINS}DATASET_{CROP}.csv', 
                 datasetfile = f'data/ANM{MOD_BINS}F{FAM_BINS}DATASET_{CROP}.csv', 
                 predicted_years = "2004 2010 2017",
                 batch_size = 16, 
                 save_checkpoint = False,
                 save_checkpoint_model = 'best-model',
                 learning_rate = 0.01,
                 max_epochs = 200,
                 lr_milestones_list = [20, 50, 600, 800,],
                 loss_func_metric = 'RMSE',
                 seed = 123456,
                 crop_name = CROP,
                 exp_name = '',            
                ):
    
        self.home_dir = home_dir
        os.chdir(self.home_dir)
        
        pl.seed_everything(seed, workers=True)
        
        print(exp_name, crop_name)
        
        if len(exp_name) == 0:
            print("exp_name is not definite")
            sys.exit(0)
        if len(crop_name) == 0:
            print("crop_name is not definite")
            sys.exit(0)
            
        self.loss_func = RMSE()
        
        ## LOSS PARSING ############################
        if loss_func_metric == 'RMSE':
            self.loss_func = RMSE()
        elif loss_func_metric == 'MAE':
            self.loss_func = MAE()
        elif loss_func_metric == 'MASE':
            self.loss_func = MASE()
        elif loss_func_metric == 'QuantileLoss':
            self.loss_func = QuantileLoss()
        elif loss_func_metric == 'Myloss':
            self.loss_func = SMAPE() + MAE()
            
        self.exp_name = exp_name
        self.crop_name = crop_name
        self.scrop = crop_name
        self.batch_size = batch_size
        
        ####### PARSE PREDICTED_YEARS INT/STR TO LIST OF STR ##############################
        print(predicted_years)
        if type(predicted_years) is str:
            self.predicted_years = predicted_years.split(' ')
        elif type(predicted_years) is int:
            self.predicted_years = [str(predicted_years),]
            print(predicted_years, self.predicted_years, str(predicted_years))
        
        self.learning_rate = learning_rate
        
        self.datasetfile = datasetfile
        
        self.lr_milestones_list = lr_milestones_list
        
        self.max_epochs = max_epochs
        
        # MOD_BINS = 512
        # FAM_BINS = 256
        
        print('predicted_years:', self.predicted_years, 
              'max_epochs:', max_epochs, 
              'batch_size:', batch_size, 
              'learning_rate', self.learning_rate, 
              'loss_func_metric:', loss_func_metric, 
              'seed:', seed, 
              'lr_milestones_list:', lr_milestones_list)
        
        # sys.exit(0)
        # fn
        
        print(f'loading {self.datasetfile}', time.asctime( time.localtime(time.time()) ) )
        alidata = pd.read_csv(self.datasetfile)
        print(f'{self.datasetfile} loaded', time.asctime( time.localtime(time.time()) ) )
        
        ######### CLEAR UNNAMED COLUMNS FROM DATASETS #######################################
        alidata = alidata.loc[:, ~alidata.columns.str.contains('^Unnamed')]

        #### SET 'county' and 'year' to categoricals, 'time_idx' to integer time step #################
        alidata['county']   = alidata['county'].astype(str)
        alidata['year']     = alidata['year'].astype(str)
        alidata['time_idx'] = alidata['time_idx'].astype(int)
        
        alidata['actuals'] = alidata[f'{self.scrop}_yield']
        
        print(type(alidata['county']), type(alidata['year']), type(alidata['time_idx'].max()))
        
        #### GET yield info and move to the end columns for view control ######################################
        yield_list = [f'{self.scrop}_sownarea', f'avg_{self.scrop}_sownarea', f'med_{self.scrop}_sownarea', \
                      f'{self.scrop}_yieldval', f'avg_{self.scrop}_yieldval', f'med_{self.scrop}_yieldval', \
                      f'{self.scrop}_yield', f'avg_{self.scrop}_yield', f'med_{self.scrop}_yield']

        cols_to_move = yield_list
        alidata = alidata[ [ col for col in alidata.columns if col not in cols_to_move ] + cols_to_move ]

        ########## DON'T DELETE, cut dataset by month for encoder/decoder length reduce ###################
        alidata = alidata[ alidata['month'] < 10 ]
        # alidata['month'] = alidata['month'].astype(str)
        
        #### ADD 'gstage' COLUMN FOR GROWTH STAGES ###################################
        alidata['gstage'] = 'yield'
        
        #### GET info alidata columns names #######
        
        alidata_list = [f'county', f'year', f'month', f'gstage', f'time_idx', f'actuals']
        
        #### GET MODIS column names #####################
        ################ MODIS cloumns name ################################
        mod_names = [f'b{iband}b{bins}' for iband in range(9) for bins in range(MOD_BINS)]      
        
        
        #### GET ONLY MODIS #########
        
        alimodis_list = []
        alimodis_list.extend(alidata_list)
        alimodis_list.extend(yield_list)
        print(alimodis_list)
        alimodis_list.extend(mod_names)
        
        
        alimodis = alidata.loc[:, alimodis_list]
        
        # alidata = alimodis
        
        # fn
        
            
################### SET INFERENCE DATAS #######################################################
        infer_mask = alidata['year'].isin(['2019', '2020', '2021', '2022'])

        data_infer = alidata[infer_mask]

        data_infer[f'{self.scrop}_sownarea'] = 0.0    #np.nan
        data_infer[f'{self.scrop}_yieldval'] = 0.0    #np.nan
        data_infer[f'{self.scrop}_yield']    = 0.0    #np.nan

        years = [str(x) for x in range(2003, 2019)]

        self.val_years = self.predicted_years 
        #### REMOVE VALIDATION YEARS FROM TRAIN DATAS #################################### 
        for iyear in self.val_years:
            print(years)
            print(iyear, type(iyear))
            years.remove(iyear)      
        self.years = years
        
        print('Years to train:', self.years)
        print('Years to valid:', self.val_years)
        
        # fn
        
        # tt = [x for x in self.val_years]
        # print('tt', tt)
        
        ##### SET TRAIN/VALIDATION DATAS #################################
        train_mask = alidata['year'].isin(self.years)
        self.data = alidata[train_mask]

        val_mask = alidata['year'].isin(self.val_years)
        self.data_val = alidata[val_mask]
        
        # delete 2008 year from dataset
        self.data = self.data[ self.data['year'] != '2008' ]
        self.years = self.data['year'].unique()
        
        print('--------check 2008----------------------')
        print('Years to train:', self.years)
        print('Years to valid:', self.val_years)
        print('Years to valid:', self.data_val['year'].unique())
        print('------------------------------')


        #### CREATE INFERENCE DATAS 2019-2023 with added validation dataset for control K-FOLD accuracy #############
        self.data_inference = pd.concat([self.data_val, data_infer], axis=0)

        #### CROP GROWTH CALENDAR CONSTRUCTOR #########################################################
        MAYDAY = 7
        HARDAY = 8
        #### CREATE TRAIN/VALIDATION/TEST DATASETS WITH ZERO/AVERAGE IN ENCODER AND GROWTH/YIELD IN DECODER ######## 
        #### SET 'gstage'='no' for encoder and growth/yield for decoder ############################
        for county in self.data['county'].unique():
            for year in self.data['year'].unique():
                avg_yield = self.data[f'avg_{self.scrop}_yield'].loc[(self.data['county'] == county) \
                                                                     & (self.data['year'] == year)].mean()
                med_yield = self.data[f'med_{self.scrop}_yield'].loc[(self.data['county'] == county) \
                                                                     & (self.data['year'] == year)].mean()
                _yield = self.data[f'{self.scrop}_yield'].loc[(self.data['county'] == county) \
                                                              & (self.data['year'] == year)].mean()
                 
                self.data[f'{self.scrop}_yield'].loc[(self.data['county'] == county) & (self.data['year'] == year) & \
                                            (self.data['month'] < MAYDAY) ] = 0.0   # avg_yield
                self.data['gstage'].loc[(self.data['county'] == county) & (self.data['year'] == year) & \
                                        (self.data['month'] < MAYDAY) ] = "no"       
                
                # self.data[f'{self.scrop}_yield'].loc[( (self.data['county'] == county) & (self.data['year'] == year) ) & \
                #                             ( (self.data['month'] == 6) | (self.data['month'] == 7) ) ] = \
                #                        [avg_yield + ((_yield - avg_yield) / 8.0) * i for i in range(1,9)]
                # self.data['gstage'].loc[( (self.data['county'] == county) & (self.data['year'] == year) ) & \
                #                         ( (self.data['month'] == 6) | (self.data['month'] == 7) ) ] = \
                #                         "growth"
                
                

        for county in self.data_val['county'].unique():
            for year in self.data_val['year'].unique():
                avg_yield = self.data_val[f'avg_{self.scrop}_yield'].loc[(self.data_val['county'] == county) \
                                                                         & (self.data_val['year'] == year)].mean()
                med_yield = self.data_val[f'med_{self.scrop}_yield'].loc[(self.data_val['county'] == county) \
                                                                         & (self.data_val['year'] == year)].mean()
                _yield = self.data_val[f'{self.scrop}_yield'].loc[(self.data_val['county'] == county) \
                                                                  & (self.data_val['year'] == year)].mean()
                
                self.data_val[f'{self.scrop}_yield'].loc[(self.data_val['county'] == county) & (self.data_val['year'] == year) & \
                                            (self.data_val['month'] < MAYDAY) ] = 0.0     # avg_yield
                self.data_val['gstage'].loc[(self.data_val['county'] == county) & (self.data_val['year'] == year) & \
                                        (self.data_val['month'] < MAYDAY) ] = "no"       
                
                # self.data_val[f'{self.scrop}_yield'].loc[( (self.data_val['county'] == county) & (self.data_val['year'] == year) ) & \
                #                             ( (self.data_val['month'] == 6) | (self.data_val['month'] == 7) ) ] = \
                #                        [avg_yield + ((_yield - avg_yield) / 8.0) * i for i in range(1,9)]
                # self.data_val['gstage'].loc[( (self.data_val['county'] == county) & (self.data_val['year'] == year) ) & \
                #                         ( (self.data_val['month'] == 6) | (self.data_val['month'] == 7) ) ] = \
                #                         "growth"

        for county in self.data_inference['county'].unique():
            for year in self.data_inference['year'].unique():
                avg_yield = self.data_inference[f'avg_{self.scrop}_yield'].loc[(self.data_inference['county'] == county) & \
                                                                      (self.data_inference['year'] == year)].mean()
                med_yield = self.data_inference[f'med_{self.scrop}_yield'].loc[(self.data_inference['county'] == county) & \
                                                                      (self.data_inference['year'] == year)].mean()
                
                self.data_inference[f'{self.scrop}_yield'].loc[(self.data_inference['county'] == county) \
                                                 & (self.data_inference['year'] == year) & \
                                                 (self.data_inference['month'] < MAYDAY) ] \
                                                 = avg_yield# (avg_yield + med_yield) / 2.0
                self.data_inference['gstage'].loc[(self.data_inference['county'] == county) \
                                                 & (self.data_inference['year'] == year) & \
                                                 (self.data_inference['month'] < MAYDAY) ] \
                                                 = "no"

                # self.data_inference[f'{self.scrop}_yield'].loc[(self.data_inference['county'] == county) \
                #                                  & (self.data_inference['year'] == year) & \
                #                                  ( (self.data_inference['month'] == 6) | (self.data_inference['month'] == 7) ) ] = \
                #                                  avg_yield
                # # [avg_yield + ((rice_yield - avg_yield) / 8.0) * i for i in range(1,9)]
                # self.data_inference['gstage'].loc[(self.data_inference['county'] == county) \
                #                                  & (self.data_inference['year'] == year) & \
                #                                  ( (self.data_inference['month'] == 6) | (self.data_inference['month'] == 7) ) ] = \
                #                                  "growth"        
                    
                    
        #### SET 'month' to the catigorical #############################
        self.data['month'] = self.data['month'].astype(str)
        self.data_val['month'] = self.data_val['month'].astype(str)
        self.data_inference['month'] = self.data_inference['month'].astype(str)
        
        
        # ADD ACTUALS FOR CONTROL  
        # self.data['actuals'] = self.data[f"{self.scrop}_yieldval"] / self.data[f"{self.scrop}_sownarea"]
        # self.data_val['actuals'] = self.data_val[f"{self.scrop}_yieldval"] / self.data_val[f"{self.scrop}_sownarea"]
        # self.data_inference['actuals'] = self.data_inference[f"{self.scrop}_yieldval"] / self.data_inference[f"{self.scrop}_sownarea"]
        
        #### ADD 'sample' COLUMN AS 'year' column & RENAME 'sample' column with INT NUMBER  ###################
        self.data_val['sample'] = self.data_val['year'].values
        
        NSAMPLES = 0
        for iyear in self.data_val['year'].unique():
            self.data_val.loc[self.data_val['year'] == iyear, 'sample'] = str(NSAMPLES)
            NSAMPLES = NSAMPLES + 1
        
        print('DATA_VAL:', self.data_val['sample'].unique(), self.data_val.shape)
        
        df = self.data_val[ (self.data_val['sample'] == self.data_val['sample'].unique()[0])]
        
        print('DATA_VAL:', self.data_val['sample'].unique(), df.shape)
        
        df = self.data_val[ (self.data_val['sample'] == self.data_val['sample'].unique()[0]) & (self.data_val['county'] == self.data_val['county'].unique()[0]) ]
        
        print('DATA_VAL:', self.data_val['sample'].unique(), df.shape)
        
####################### INIT DATA_TRAIN appears and add data_val as last year to each samples##########################
        self.data_train, _ = DataGenerator3(DATA=self.data, 
                                            YEARS_MAX_LENGTH=5,
                                            NSAMPLES=len(self.data_val['sample'].unique()))
        
        ###### ADD VALIDATION TILE TO TRAIN DATA FOR CUT IT IN VALIDATION DATALOADER IN PREDICTED MODE #############
        ##### WITH time_idx recalculation #########################
        df_tr = pd.DataFrame()
        for smpl in self.data_val['sample'].unique():
            for county in self.data_val['county'].unique():
                df_cn = pd.DataFrame()
                dfa = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == county) ]
                dfb = self.data_val[ (self.data_val['sample'] == smpl) & (self.data_val['county'] == county) ]
                df_cn = pd.concat([df_cn, dfa], axis=0)
                df_cn = pd.concat([df_cn, dfb], axis=0)
                
                new_index = pd.RangeIndex(start=0, stop=len(df_cn)+0, step=1)
                df_cn.index = new_index
                df_cn["time_idx"] = df_cn.index.astype(int)
                df_tr = pd.concat([df_tr, df_cn], axis=0)
        new_index = pd.RangeIndex(start=0, stop=len(df_tr)+0, step=1)
        df_tr.index = new_index
        
        self.data_train = df_tr
        
        ########### PLOT SAMPLE WITH ENCODER/DECODER FOR CONTROL ################################################

        smpl = self.data_train['sample'].unique()[0] #self.val_years[0]
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
        # fig, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))
        
        print('type(self.data_train[time_idx])', type(self.data_train['time_idx']))
        
        df = self.data_train[ (self.data_train['county'] == '0') & (self.data_train['sample'] == smpl) ]
        
        print('time_idx', df['time_idx'].to_numpy())
        print('sample', self.data_train['sample'].unique())
        print('county', self.data_train['county'].unique())
        print('df[year].unique()', df['year'].unique())
        
        print('df[time_idx].to_numpy()', df['time_idx'].to_numpy())
        # print(dfali['time_idx'].to_numpy())
        
        ax.plot(df['time_idx'].to_numpy(), df[f'{self.scrop}_yield'].to_numpy(), 'o')
        # Create the second y-axis
        ax2 = ax.twiny().twinx()
        ax2.plot(df['time_idx'].to_numpy(), df['gstage'], 'x', color='green')

        ######### SET ENCODER-DECODER LENGTH AS 'gstage' phase: no/growth/yied ####################################################
        
        dfsmpl = self.data_train[ (self.data_train['sample'] == smpl)]
        print('self.val_years[0]:',self.val_years, dfsmpl['year'].unique())
        
        dfsmpl = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') ]
        print('self.val_years[0]:',self.val_years[0], dfsmpl['year'].unique())
        
        df_ = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') ]
        # ival_years = self.val_years.copy()
        ival_years = [df_['year'].unique()[-1],]
        for ii in range(3-len(ival_years)):
            ival_years.append(ival_years[0])
        
        print('ival_years:', ival_years)
                       
            
        
#         dflast = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0')  & ( (self.data_train['year'] == self.val_years[0]) | (self.data_train['year'] == self.val_years[1]) | (self.data_train['year'] == self.val_years[2]) )]
        
#         dfe = self.data_train[ (self.data_train['gstage'] == 'no') & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') & ( (self.data_train['year'] == self.val_years[0]) | (self.data_train['year'] == self.val_years[1]) | (self.data_train['year'] == self.val_years[2]) )] 
        
#         dfp = self.data_train[ ( (self.data_train['gstage'] == 'growth') | (self.data_train['gstage'] == 'yield') ) & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0')  & ( (self.data_train['year'] == self.val_years[0]) | (self.data_train['year'] == self.val_years[1]) | (self.data_train['year'] == self.val_years[2]) )]

        dflast = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0')  & ( (self.data_train['year'] == ival_years[0]) | (self.data_train['year'] == ival_years[1]) | (self.data_train['year'] == ival_years[2]) )]
        
        dfe = self.data_train[ (self.data_train['gstage'] == 'no') & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') & ( (self.data_train['year'] == ival_years[0]) | (self.data_train['year'] == ival_years[1]) | (self.data_train['year'] == ival_years[2]) )] 
        
        dfp = self.data_train[ ( (self.data_train['gstage'] == 'growth') | (self.data_train['gstage'] == 'yield') ) & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0')  & ( (self.data_train['year'] == ival_years[0]) | (self.data_train['year'] == ival_years[1]) | (self.data_train['year'] == ival_years[2]) )]

        
#         dflast = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') ]
        
#         dfe = self.data_train[ (self.data_train['gstage'] == 'no') & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') ] 
        
#         dfp = self.data_train[ ( (self.data_train['gstage'] == 'growth') | (self.data_train['gstage'] == 'yield') ) & (self.data_train['sample'] == smpl) & (self.data_train['county'] == '0') ]
        
        ################!!!!!! SET max_prediction_length & max_encoder_length value !!!!!!!#######
        self.max_prediction_length = dfp.shape[0]
        self.max_encoder_length    = dfe.shape[0]
        
        ######## PLOT & CHECK ENCODER/DECODER DATASET ############################################################
        
        ax.plot(dfe['time_idx'].to_numpy(), dfe[f'{self.scrop}_yield'].to_numpy(), '.', color='yellow')
        ax.plot(dfp['time_idx'].to_numpy(), dfp[f'{self.scrop}_yield'].to_numpy(), '.', color='red')
        
        print(self.max_encoder_length, self.max_prediction_length)   
        
        last_year = dfe['year'].unique()
        print('last_year:', last_year)
        dfali = alidata[ (alidata['county'] == '0') & (alidata['year'] == last_year[0]) ]
        print(len(dfali['time_idx'].to_numpy()), dfali['time_idx'].to_numpy())
        print(len(dflast['time_idx'].to_numpy()), dflast['time_idx'].to_numpy())
        dfali['time_idx'] = dflast['time_idx'].values
        print(dfali['time_idx'].to_numpy())
        ax.plot(dfali['time_idx'].to_numpy(), dfali[f'{self.scrop}_yield'].to_numpy(), '-.')
        
        plt.show()
        plt.savefig(f'A0F_{self.exp_name}', bbox_inches='tight')           
        
        # fn
        
        # self.data_train = self.data
        
        # get the memory usage in bytes
        memory_bytes = self.data_train.memory_usage().sum()

        # convert to megabytes
        memory_mb = memory_bytes / (1024 ** 2)

        print(f"Dataframe size: {memory_mb:.2f} Mb")
        
        print('DataGenerator done...')
        
        ###### SET BASIC FILENAME #######################################
        self.name_for_files = f'EXP_[{self.exp_name}]-Cr[{self.scrop}]-KF[{"_".join(self.val_years)}]-BS[{self.batch_size}]'
        if os.path.exists(self.name_for_files) == True:
            print(f'Experiment exist: {self.name_for_files}')
            print(f'Set another exp_name...')
            # sys.exit(0)
        
        print('Set basic filenames self.name_for_files:', self.name_for_files)
        
        # fn
        
        ##### CHECK data_train #####################################
        df_ = self.data_train #[(self.data_train['gstage'] == 'yield')]
        y_true1 = df_[f'{self.scrop}_yield'].to_numpy()
        df_ = self.data_train[(self.data_train['gstage'] == 'no')]
        y_true2 = df_[f'{self.scrop}_yield'].to_numpy()
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_true1, 'o', color='green', label='actuals')
        ax.plot(y_true2, '.', color='blue', label='actuals')
        # ax.plot(y_pred.cpu().numpy(), '.', color='red', label='predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        # ax.set_title(f'Validation SMAPE: {smape_val:.2%}')
        ax.legend()
        
        # save the plot as an image
        plt.savefig(f"A00001_{self.exp_name}.png")
        
        # fn
        ####################################################################
        
        # fn
        
            
########## SET ENCODER/DECODER COVARIATES #######################################################
        # avg_med = ["avg_rice_yield", "med_rice_yield", "avg_rice_sownarea", "med_rice_sownarea",\
        #                  "avg_rice_yieldval", "med_rice_yieldval"]
        
        # avg_med = ["avg_rice_yield", "rice_sownarea"]
        
        # avg_med = [f"avg_{self.scrop}_yield", f"actuals"]
        
        avg_med = [f"avg_{self.scrop}_yield", 
                   f"avg_{self.scrop}_sownarea", 
                   # f"avg_{self.scrop}_yieldval", 
                   f"{self.scrop}_sownarea", 
                   # f"actuals",
                  ]
        
        avg_med = []
        
        # avg_med = [f"avg_{self.scrop}_yield"]

        _static_reals = avg_med
        
        print("avg_med:", avg_med)
        
        time.sleep(15)      

        # display( data[ [ col for col in _time_varying_known_reals ] ] )

        # fn

        ################ MODIS cloumns name ################################
        modis_list = [f'b{iband}b{bins}' for iband in range(9) for bins in range(MOD_BINS)]

        ################ FAMINA cloumns name ################################
        
        famine_list = ['Evap_tavg', 
                       'LWdown_f_tavg', 'Lwnet_tavg', 'Psurf_f_tavg', \
                       'Qair_f_tavg', 'Qg_tavg',\
                       'Qh_tavg', 'Qle_tavg', 'Qs_tavg', 'Qsb_tavg', \
                       'RadT_tavg', 'Rainf_f_tavg', \
                       'SnowCover_inst', 'SnowDepth_inst', 'Snowf_tavg', \
                       'SoilMoi00_10cm_tavg', 'SoilMoi10_40cm_tavg', \
                       'SoilMoi40_100cm_tavg', \
                       'SoilTemp00_10cm_tavg', 'SoilTemp10_40cm_tavg', \
                       'SoilTemp40_100cm_tavg', \
                       'SWdown_f_tavg', 'SWE_inst', 'Swnet_tavg', 'Tair_f_tavg', 'Wind_f_tavg',
                      ]

        nbins = ['_' + str(x) for x in range(0, FAM_BINS - 1)]

        famine_names = [famine + bb for famine in famine_list for bb in nbins]
        
        
############# PREPARE variables for the TimeSeriesDataSet  ###################################
        self._time_varying_known_reals = []
        self._time_varying_known_reals.extend(avg_med)
        self._time_varying_known_reals.extend(modis_list) 
        self._time_varying_known_reals.extend(famine_names)

        self._time_varying_unknown_reals = []
        self._time_varying_unknown_reals.extend(avg_med)
        self._time_varying_unknown_reals.extend(modis_list)
        self._time_varying_unknown_reals.extend(famine_names)

        # print( self.data.sort_values("time_idx").groupby(["county", "year"]).time_idx.diff().dropna() == 1 )    
        
#####################################################################################################################
############################## SET TRAIN/VALIDATION/TEST TS DATASETS ###################################################

        #### ADD TIME LAG TO ENCODER/DECODER #################################################### 
        self.prediction_lag = 0
        
        self.training = TimeSeriesDataSet(
            self.data_train[lambda x: x.time_idx <= x.time_idx.max() - self.max_prediction_length - self.prediction_lag],
            # self.data_train,
            # self.data_val,
            time_idx="time_idx",
            target=f"{self.scrop}_yield",
            group_ids=["county", "sample"],
            # group_ids=["county", "year"],
            # min_encoder_length=self.max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
            max_encoder_length = self.max_encoder_length - self.prediction_lag,
            # min_prediction_length = 2,                     #max_prediction_length // 2,
            max_prediction_length = self.max_prediction_length + self.prediction_lag,
            # min_prediction_idx = min_prediction_idx,
            # static_categoricals = ["county", "year"],
            # static_reals = _static_reals,
            time_varying_known_categoricals=["month", "gstage"],
            # variable_groups={"years": years},  # group of categorical variables can be treated as one variable
            time_varying_known_reals = self._time_varying_known_reals,
            # time_varying_unknown_categoricals=[],
            time_varying_unknown_reals = self._time_varying_unknown_reals,
            # target_normalizer=GroupNormalizer(
            #     groups=["county", "sample"], #transformation="relu"
            # ),  # use softplus and normalize by group
            target_normalizer=TorchNormalizer(
                # method = "standard",  
                method = "identity",
            ),  # use softplus and normalize by group
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

# #         print( time.asctime( time.localtime(time.time()) ) )
        

####### CREATE VALIDATION/TEST TSDS (predict=True) which means to predict the last max_prediction_length points in time
#       # for each series
        self.validation = TimeSeriesDataSet.from_dataset(self.training, 
                                                         self.data_train, 
                                                         # self.data_val,
                                                         predict=True, 
                                                         stop_randomization=True)
        
        self.testing = TimeSeriesDataSet.from_dataset(self.training, 
                                                      self.data_train, 
                                                      # predict=True, 
                                                      stop_randomization=True)

#         print(f'training & validation TimeSeriesDataSet loaded', time.asctime( time.localtime(time.time()) ) )
        
###################### CREATE TRAIN/VALIDATION/TEST DATALOADERS FROM TimeSeriesDataSet################################
#         # batch_size = 16  # set this between 32 to 128
        self.train_dataloader = self.training.to_dataloader(train=True, 
                                                            batch_size=self.batch_size, 
                                                            num_workers=8)
        
        self.val_dataloader = self.validation.to_dataloader(train=False, 
                                                            # batch_size=27, 
                                                            batch_size=30,
                                                            num_workers=8)
        
        self.test_dataloader = self.training.to_dataloader(train=False, 
                                                           batch_size=30, 
                                                            # batch_size=self.batch_size, 
                                                           num_workers=8)
        
        # self.test_dataloader = self.testing.to_dataloader(train=False, 
        #                                                   # batch_size=27, 
        #                                                   batch_size=30,
        #                                                   num_workers=8)
        
        print('Dataloaders len:', len(self.train_dataloader), len(self.val_dataloader), len(self.test_dataloader))
        
        # sys.exit(0)
        
#         ##### CHECK train_dataloader #####################################
#         # y_true = torch.cat([y[0] for x, y in iter(self.train_dataloader)])
#         y_true1 = torch.cat([y for x, (y, weight) in iter(self.train_dataloader)])
#         y_true2 = torch.cat([y for x, (y, weight) in iter(self.val_dataloader)])
#         # Create plot
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(y_true1.cpu().numpy(), 'o', color='green', label='actuals')
#         ax.plot(y_true2.cpu().numpy(), '.', color='red', label='actuals')
#         # ax.plot(y_pred.cpu().numpy(), '.', color='red', label='predictions')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Value')
#         # ax.set_title(f'Validation SMAPE: {smape_val:.2%}')
#         ax.legend()
        
#         # save the plot as an image
#         plt.savefig(f"A00002.png")
#         # plt.savefig(f"{self.filename}.png")
        
#         # fn
#         ####################################################################
        
#         print('self.train_dataloader:', len(self.train_dataloader))
#         print('self.val_dataloader:', len(self.val_dataloader))
#         print('self.test_dataloader:', len(self.test_dataloader))
        
#         # fn
        
#         print( time.asctime( time.localtime(time.time()) ) )

#         #### TEST BASELINE ##########################################################
#         # actuals = torch.cat([y for x, (y, weight) in iter(self.val_dataloader)])
#         # baseline_predictions = Baseline().predict(self.val_dataloader)
#         # calculate baseline mean absolute error, i.e. predict next value as the last available value from the history
#         baseline_predictions = Baseline().predict(self.val_dataloader, return_y=True)
#         actuals = baseline_predictions.y
#         MAE()(baseline_predictions.output, baseline_predictions.y)
#         print( 'Baseline:', MAE()(baseline_predictions.output, baseline_predictions.y) )
#         print( 'Baseline:', type(actuals), type(baseline_predictions.output), type(baseline_predictions.y) )
#         print( 'Baseline:', baseline_predictions.y )
#         print( 'Baseline:', (actuals[0] - baseline_predictions.output).abs().mean() )
#         print( 'Baseline:', time.asctime( time.localtime(time.time()) ) )
            
        # dir = '/hy-tmp/chck/ali'
        # home_dir = '/content/gdrive/My Drive/AChina' 
        # _dir = os.path.join(home_dir, 'data')
        
############ SET EXPERIMENT SETTINGS FOR TRAINER #############################################################        
        #### SET CHECKPOINT ##############################
        reseter_step = 5
        self.ModelCheckpointPath = os.path.join(home_dir, self.name_for_files)
        self._checkpoint_callback = ModelCheckpoint(dirpath = self.ModelCheckpointPath, every_n_epochs = reseter_step)
        
        self._Reseter = Reseter(ModelCheckpointPath = self.ModelCheckpointPath, milestones = reseter_step)

        _dir = '/tf_logs'
        # dir = os.path.join(home_dir, 'data')
        
        ##### SET TENSORBOARD ############################################
        self._tb_logger = TensorBoardLogger(_dir, name = self.name_for_files, comment = self.name_for_files)
        
        milstones_list = [x for x in range(0, self.max_epochs, 25)]
        
#         _actvspred_train = ActualVsPredictedCallback(self.train_dataloader, 
#                                                filename = f'{self.name_for_files}_train', 
#                                                milestones = milstones_list)
        
#         _actvspred_valid = ActualVsPredictedCallback(self.val_dataloader, 
#                                                filename = f'{self.name_for_files}_valid', 
#                                                milestones = milstones_list)
        
        # _actvspred_train = CustomTrainingLogger()

        #### SEL LEARNING RATE MONITOR ###################################
        self._lr_monitor = LearningRateMonitor(logging_interval = 'epoch')

        #### LEARNING RATE TUNER #########################################
        self.learning_rate = 0.05
        
        # self._lr_finder  = FineTuneLearningRateFinder_CyclicLR2(base_lr=self.learning_rate, 
        #                                                         max_lr=0.05, 
        #                                                         step_size_up=100, 
        #                                                         step_size_down=100,
        #                                                         mode='triangular2')     
        
        # self._lr_finder = FineTuneLearningRateFinder_CustomLR(total_const_iters=5, 
        #                                                       base_lr=self.learning_rate, 
        #                                                       max_lr=0.05, 
        #                                                       step_size_up=250, 
        #                                                       step_size_down=20050, 
        #                                                       cycle_iters=2,
        #                                                       mode='triangular',) 
        
        self._lr_finder = FineTuneLearningRateFinder_MultiStepLR(milestones=[900, 1000])
        
        # self._lr_finder = FineTuneLearningRateFinder_LinearLR(total_iters=75)
        
        # self._lr_finder = FineTuneLearningRateFinder_CustomLR2(constant_iters=10, 
        #                                                        linear_iters=15, 
        #                                                        linear_pleutau_iters=15,
        #                                                        step_size=10,)    
        
        #### GRADIENT ACCUMULATION SHEDULER ####################################
        _GradAccumulator = GradientAccumulationScheduler(scheduling={0: 4, 60: 4, 150: 4})
        
        #### STOCHASTIC WEIGHT AVERAGIN #######################################
        _SWA = StochasticWeightAveraging(swa_lrs=1e-2, 
                                         swa_epoch_start=50, 
                                         device='gpu')
        
#         #### RELOAD DATALOADER EVERY EPOCHS ########################################
#         _reload_dataloader = ReloadDataLoader(self.training, self.batch_size)
        
#         #### RELOAD TRAINING DATASET AND DATALOADER EVERY EPOCHS ###################
#         _reload_dataset = ReloadDataSet_12(self.data, 
#                                            self.training, 
#                                            self.train_dataloader,
#                                            self.batch_size, 
#                                            YEARS_MAX_LENGTH=10, 
#                                            NSAMPLES=len(self.data_val['sample'].unique()))

################## SET TRAINER ###########################################################
        self.trainer = Trainer(accelerator = 'gpu', 
                               logger = self._tb_logger, 
                               log_every_n_steps = 1, 
                               max_epochs = self.max_epochs,
                               # devices = "0",          
                               # fast_dev_run=True, 
                               # precision=16,
                               # gradient_clip_val = 0.2,
                               reload_dataloaders_every_n_epochs = 1,
                               callbacks = [self._lr_monitor,
                                            # self._lr_finder, 
                                            self._checkpoint_callback,                                      
                                            # _reload_dataset, 
                                            # # _tb_logger, in logger
                                            # _actvspred_train, 
                                            # _actvspred_valid,
                                            self._Reseter,
                                            ])
        


############## SET TEMPORAL FUSION TRANSFORMER AS MODEL #######################################

        self.tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=self.learning_rate,
            # lstm_layers=2,
            # hidden_size=42,             # most important hyperparameter apart from learning rate
            # hidden_continuous_size=30,  # set to <= hidden_size
            # attention_head_size=4,      # number of attention heads. Set to up to 4 for large datasets
            # dropout=0.3,           
            # output_size=7,  # 7 quantiles by default      
            loss=self.loss_func,
            # loss=QuantileLoss(),
            # optimizer = 'adam',
            optimizer = 'sgd',
            # log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            # reduce_on_plateau_patience=4,
            )

######## SET best_tft and checkpont for inferencing at the end of the train(self,) ####################################
        
        self.best_tft = self.tft
        self.checkpoint = self.name_for_files 
        
        self.gif = 0

################## THE FIN __init__ #######################################################################
################## THE MODELS FUNCTIONS ############################################
    ####### PREDICT AND PLOT @DATALOADER #####################      
    def pltprd(self, dataloader, prfx=''):
        baseline_predictions = self.tft.predict(dataloader, return_y=True)
        actuals = baseline_predictions.y
        # MAE()(baseline_predictions.output, baseline_predictions.y)
        print('pltprd:', len(dataloader), len(actuals[0]), len(baseline_predictions.output), baseline_predictions.keys()) #, actuals[0])
        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        raw_predictions = self.tft.predict(dataloader, mode="raw", return_x=True, return_y=True)
        # print(type(raw_predictions.y), raw_predictions.y)
        print(type(raw_predictions.output), type(raw_predictions.output[0]))
        print('MAE:', MAE()(baseline_predictions.output, baseline_predictions.y) )
        print('raw_predictions:', raw_predictions.keys(), len(raw_predictions.output), len(raw_predictions.y[0]))
        # print(baseline_predictions.y)
        # print(baseline_predictions.output)
        # fn
        y_true = actuals[0]
        y_pred = baseline_predictions.output
        
        # avg_yield = self.data_val[f'avg_{self.scrop}_yield'].loc[(self.data_val['county'] == county) \
        #                           & (self.data_val['year'] == year)].mean()
        # y_avg = f'avg_{self.scrop}_yield'
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_true.cpu().numpy(), 
                'o', 
                color='green', )
                # label='actuals')
        ax.plot(y_pred.cpu().numpy(), 
                '.', 
                color='red', )
                # label='predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'frame={self.gif}')
        ax.legend()
        
        ax.set_ylim([0, 1])

        # print('ActPred3', y_true.device, y_pred.device)

        # save the plot as an image
        plt.savefig(f"{prfx}_{self.name_for_files}_actuals_vs_predictions.png",
                   transparent = False,
                   facecolor = 'white'
                   )
            
        plt.savefig(f"./imgA/{prfx}_{self.name_for_files}_actuals_vs_predictions_{self.gif}.png",
                   transparent = False,
                   facecolor = 'white'
                   )
        
        self.gif = self.gif + 1
        
        plt.close()
        
        # sys.exit(0)
        
    ### TRAIN TFT MODEL ##################################################    
    def train(self,):
        print( time.asctime( time.localtime(time.time()) ) )    
        # initialize a list to hold the .ckpt files
        for iepoch in range(self.max_epochs):
            
            ckpt_files = []
            try:
                # get a list of all the files in the parent directory with a .ckpt extension
                ckpt_files = [f for f in os.listdir(self.ModelCheckpointPath) if f.endswith('.ckpt')]
            except FileNotFoundError:
                # handle the case where the parent directory doesn't exist
                print("No checkpoint found, maybe it's first start")
            print(ckpt_files)
        
            if len(ckpt_files) == 0:                

                self.trainer.fit(
                    self.tft,
                    train_dataloaders = self.train_dataloader,
                    val_dataloaders   = self.val_dataloader,
                )
                # self.pltprd(self.val_dataloader)
                # self.trainer.should_stop = False

            else:
#                 self.data_train, _ = DataGenerator2(DATA=self.data, 
#                                                     YEARS_MAX_LENGTH=5,
#                                                     NSAMPLES=len(self.data_val['sample'].unique()))
                
#                 df_tr = pd.DataFrame()
#                 for smpl in self.data_val['sample'].unique():
#                     for county in self.data_val['county'].unique():
#                         df_cn = pd.DataFrame()
#                         dfa = self.data_train[ (self.data_train['sample'] == smpl) & (self.data_train['county'] == county) ]
#                         dfb = self.data_val[ (self.data_val['sample'] == smpl) & (self.data_val['county'] == county) ]
#                         df_cn = pd.concat([df_cn, dfa], axis=0)
#                         df_cn = pd.concat([df_cn, dfb], axis=0)

#                         new_index = pd.RangeIndex(start=0, stop=len(df_cn)+0, step=1)
#                         df_cn.index = new_index
#                         df_cn["time_idx"] = df_cn.index.astype(int)
#                         df_tr = pd.concat([df_tr, df_cn], axis=0)
#                 new_index = pd.RangeIndex(start=0, stop=len(df_tr)+0, step=1)
#                 df_tr.index = new_index

#                 self.data_train = df_tr

#                 self.dataset_train = TimeSeriesDataSet.from_dataset(self.training, 
#                                                                     self.data_train[lambda x: x.time_idx <= x.time_idx.max() - self.max_prediction_length - self.prediction_lag],)
#                                                                     # self.data_train)

#                 self.train_dataloader = self.dataset_train.to_dataloader(train=True, 
#                                                                          batch_size=self.batch_size, 
#                                                                          shuffle=True, 
#                                                                          num_workers=10)
                
#                 self.testing = TimeSeriesDataSet.from_dataset(self.training, 
#                                                               self.data_train, 
#                                                               # predict=True, 
#                                                               stop_randomization=True)

#                 self.test_dataloader = self.testing.to_dataloader(train=False, 
#                                                                   batch_size=27, 
#                                                                   num_workers=8)

                self.trainer.fit(
                    self.tft,
                    train_dataloaders = self.train_dataloader,
                    val_dataloaders   = self.val_dataloader,
                    ckpt_path=f"{self.ModelCheckpointPath}/{ckpt_files[0]}",
                )
                
            ########## for set trainer to the fit mode #########
            self.trainer.should_stop = False
            
            # self.pltprd(self.test_dataloader, prfx='test')
            self.pltprd(self.val_dataloader, prfx='valid')
            
            # sys.exit(0)
                
            iepoch = self.trainer.current_epoch
            print('train epoch:', iepoch)         
                
        print('fit:', time.asctime( time.localtime(time.time()) ) )
        
        # load the best model according to the validation loss
        # (given that we use early stopping, this is not necessarily the last epoch)
        # os.chdir(os.path.join(home_dir, 'data'))
        #   best_model_path = trainer.checkpoint_callback.best_model_path
        # print(type(best_model_path))
        # best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        print(f'weights saving to checkpoint: {self.checkpoint} in:', time.asctime( time.localtime(time.time()) ) )
        # checkpoint = f"{self.crop}-{self.val_year}-{self.exp_name}.ckpt"
        self.trainer.save_checkpoint(f'{self.checkpoint}.ckpt')
        print('weights loading', time.asctime( time.localtime(time.time()) ) )
        self.best_tft = self.tft  # TemporalFusionTransformer.load_from_checkpoint(checkpoint)
        print('weights loaded', time.asctime( time.localtime(time.time()) ) )
        
        # print('Train(): learning_rate', self.model.hparams.learning_rate)
        # self.trainer = self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        # self.best_tft = self.model
        # if self.save_checkpoint_model == 'best-model':
        #     print(f"{self.crop_name} {self.save_checkpoint} best-model loading...")
        #     best_model_path = self.trainer.checkpoint_callback.best_model_path
        #     print('best_model_path:', type(best_model_path), best_model_path)
        #     self.best_tft = TemporalFusionTransformer.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path)
        #     print(f"{self.crop_name} {self.save_checkpoint} best-model loaded...")
        #     if self.save_checkpoint == True:
        #         self.best_tft.save_checkpoint(self.checkpoint)
        #         print(f"{self.crop_name} {self.save_checkpoint} best-model saved...")
        # elif self.save_checkpoint_model == 'last-model':
        #     self.trainer.save_checkpoint(self.checkpoint)
        #     self.best_tft = TemporalFusionTransformer.load_from_checkpoint(self.checkpoint)
        #     print(f"{self.crop_name} {self.save_checkpoint} last-model loaded...")
    
    ### PREDICT ##################################################################
    def predict(self,):
        print('predict')
        # calcualte mean absolute error on validation set
        # actuals = torch.cat([y[0] for x, y in iter(self.val_dataloader)])
        predictions = self.best_tft.predict(self.val_dataloader)
        # (actuals - predictions).abs().mean()
        
        print('raw predict')
        
        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        raw_predictions = self.best_tft.predict(self.val_dataloader, mode="raw", return_x=True)
        
        print(type(raw_predictions), raw_predictions.keys()) 
        # print(type(x), x.keys()) 
        # print(type(raw_predictions['prediction']), raw_predictions['prediction'].shape)
        # for idx in range(27):  # plot 10 examples
        #     self.best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
            
#         import json

#         experiment = {}
#         experiment.update( raw_predictions )
#         experiment.update( x )

#         print(experiment.keys())
#         print(experiment['prediction'].numpy().shape)
#         print(experiment['encoder_target'].size())
#         print(experiment['decoder_target'].size())

#         np.savez(
#             f'{self.name_for_files}_predict.npz',
#             actuals = np.asarray(actuals), 
#             predictions = np.asarray(predictions),
#             prediction = experiment['prediction'].numpy(),
#             encoder_target = experiment['encoder_target'].numpy(),
#             decoder_target = experiment['decoder_target'].numpy(),
#             )   
        
#         print('predict saved')
        
    def test(self, checkpoit_file=f'checkpoint.ckpt'):
        print('test')
        
        self.checkpoint = checkpoit_file
        
        print(f'weights loading from checkpoint: {self.checkpoint}', time.asctime( time.localtime(time.time()) ) )
        self.best_tft = TemporalFusionTransformer.load_from_checkpoint(f'{self.checkpoint}.ckpt')
        print('weights loaded', time.asctime( time.localtime(time.time()) ) )
        
        # calcualte mean absolute error on validation set
        actuals = torch.cat([y[0] for x, y in iter(self.test_dataloader)])
        predictions = self.best_tft.predict(self.test_dataloader)
        (actuals - predictions).abs().mean()
        
        raw_predictions, x = self.best_tft.predict(self.test_dataloader, mode="raw", return_x=True)
        
        print(type(raw_predictions), raw_predictions.keys()) 
        print(type(x), x.keys()) 
        print(type(raw_predictions['prediction']), raw_predictions['prediction'].shape)
        for idx in range(81):  # plot 10 examples
            self.best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
            
        import json

        experiment = {}
        experiment.update( raw_predictions )
        experiment.update( x )

        print(experiment.keys())
        print(experiment['prediction'].numpy().shape)
        print(experiment['encoder_target'].size())
        print(experiment['decoder_target'].size())

        np.savez(
            f'{self.name_for_files}_test.npz',
            actuals = np.asarray(actuals), 
            predictions = np.asarray(predictions),
            prediction = experiment['prediction'].numpy(),
            encoder_target = experiment['encoder_target'].numpy(),
            decoder_target = experiment['decoder_target'].numpy(),
            )  
        
        print('test saved')
        
    def inference(self,):
        inference = TimeSeriesDataSet(
        self.data_inference,
        time_idx="time_idx",
        target=f"{self.scrop}_yield",
        group_ids=["county", "year"],
        min_encoder_length=self.max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length = self.max_encoder_length,
        min_prediction_length = 1 , # max_prediction_length // 2,
        max_prediction_length = self.max_prediction_length,
        # min_prediction_idx = min_prediction_idx,
        # static_categoricals = ["county", "year"],
        # static_reals = _static_reals,
        time_varying_known_categoricals=["month"],
        # variable_groups={"years": years},  # group of categorical variables can be treated as one variable
        time_varying_known_reals = self._time_varying_known_reals,
        # time_varying_unknown_categoricals=[],
        time_varying_unknown_reals = self._time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
            groups=["county", "year"], transformation="relu"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        )
        
        inf_dataloader = inference.to_dataloader(train=False, batch_size=self.batch_size, num_workers=8)
        
        # print('weights saving', time.asctime( time.localtime(time.time()) ) )
        # checkpoint = f"{scrop}_{val_year}.ckpt"
        # trainer.save_checkpoint(checkpoint)
        print(f'inference {self.checkpoint} weights loading', time.asctime( time.localtime(time.time()) ) )
        self.best_tft = TemporalFusionTransformer.load_from_checkpoint(f'{self.checkpoint}.ckpt')
        print('weights loaded', time.asctime( time.localtime(time.time()) ) )
        
        actuals = torch.cat([y[0] for x, y in iter(inf_dataloader)])
        predictions = self.best_tft.predict(inf_dataloader)
        
        # raw predictions are a dictionary from which all kind of information including quantiles can be extracted
        raw_predictions, x = self.best_tft.predict(inf_dataloader, mode="raw", return_x=True)
        
        print(type(raw_predictions), raw_predictions.keys()) 
        print(type(x), x.keys()) 
        print(type(raw_predictions['prediction']), raw_predictions['prediction'].shape)
        
        for idx in range(27):  # plot 10 examples
            self.best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);
            
        import json

        experiment = {}
        experiment.update( raw_predictions )
        experiment.update( x )

        print(experiment.keys())
        print(experiment['prediction'].numpy().shape)
        print(experiment['encoder_target'].size())
        print(experiment['decoder_target'].size())

        np.savez(
            f'{self.name_for_files}_inference.npz',
            actuals = np.asarray(actuals), 
            predictions = np.asarray(predictions),
            prediction = experiment['prediction'].numpy(),
            encoder_target = experiment['encoder_target'].numpy(),
            decoder_target = experiment['decoder_target'].numpy(),
            )  

    def plot_predict(self,):
        
        X = [X for X in range(0, self.actuals.shape[0])]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        ax1.plot(X, self.actuals, color='b', label="Actual")
        ax1.plot(X, self.predictions, color='r', label="Predicted")
        ax1.set_title(self.logger_name)

        files = os.path.join(self.home_dir, f'TFTC_{self.crop_name}_{self.predicted_years}_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        # plt.show()
    
        X = [X for X in range(1, 21)]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        outs = int( self.actuals.shape[0] / 20 )

        act = []
        pred = []
        for ii in range(0,outs*20,outs):
            act.append(self.actuals[ii:ii+outs].mean())
            pred.append(self.predictions[ii:ii+outs].mean())

        ax1.plot(X, np.asarray(act), 'bo', label="Actual")
        ax1.plot(X, np.asarray(pred), 'ro', label="Predicted")
        leg = plt.legend(loc='upper center')
        plt.xticks(X)
        ax1.set_ylim([0, 1])
        plt.xlabel("counties")
        plt.ylabel("Yield")
        ax1.set_title(f"Corn yield predictions for {self.predicted_years} with Temporal Fusion Transformer")

        files = os.path.join(self.home_dir, f'TFT_{self.crop_name}_{self.predicted_years}_yield_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        # plt.show()

        X = [X for X in range(0, actuals.shape[0])]
        X = [X for X in range(1, 21)]

        fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(20,5))

        act = []
        pred = []
        for ii in range(0,outs*20,outs):
            act.append(self.actuals[ii:ii+outs].mean())
            pred.append(self.predictions[ii:ii+outs].mean())

        ax1.plot(X, (1-np.abs(np.asarray(act)-np.asarray(pred))) * 100, 'bo', label="Actual")
        # ax1.plot(X, np.asarray(pred), 'r.', label="Predicted")
        ax1.set_ylim([70, 105])
        plt.xticks(X)
        plt.xlabel("counties")
        plt.ylabel("Yild Accuracy")
        ax1.set_title(f"ACCURACY for Temporal Fusion Transformer for {self.predicted_years} year for corn yield predict") 

        files = os.path.join(self.home_dir, f'TFT_{self.crop_name}_{self.predicted_years}_accuracy_{self.exp_name}.png')
        plt.savefig(files, bbox_inches='tight')
        
import sys
import imageio
 
class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.console.flush()
        self.file.flush()
 
    def flush(self):
        # self.console.flush()
        # self.file.flush()
        pass

class RunTask:
    @staticmethod
    def train_TFT(exp_name, 
                  crop_name='rice', 
                  predicted_years="2004 2010 2017",
                  batch_size=16,
                  learning_rate=0.0325,
                  loss_func_metric='RMSE', 
                  max_epochs=100):
        
        # print('predicted year:', predicted_years, type(predicted_years))
        
        # torch.set_float32_matmul_precision('medium')
        
        log_file = os.path.join('/hy-tmp', f'Log-cr[{crop_name}]-yr[{predicted_years}]-en[{exp_name}]-bs[{batch_size}].log')
        
#         if os.path.exists(log_file):
#             print(f'log file {log_file} exist')
#             sys.exit(0)
           
#         sys.stdout = Logger(log_file)
        
        model = ModelBase(exp_name=exp_name, 
                          predicted_years=predicted_years,
                          max_epochs=max_epochs, 
                          batch_size=batch_size, 
                          learning_rate=learning_rate,
                          loss_func_metric=loss_func_metric)
        
        # model.init_lr_finder()
        # model.custom_finder()
        # model.train()
        
        #### CREATE GIF WITH VALIDATION PREDICT MOOVEMENTS THROUGHT TRAINING CONVERGING PROCESS ######################## 
        time = [x for x in range(329)]
        print('CREATE GIFF')
        prfx = 'valid'
        frames = []
        for t in time:
            print(f"./imgA/{prfx}_{model.name_for_files}_actuals_vs_predictions_{t}.png")
            image = imageio.v2.imread(f"./imgA/{prfx}_{model.name_for_files}_actuals_vs_predictions_{t}.png")
            frames.append(image)
        
        
        imageio.mimsave(f'./{prfx}_{model.name_for_files}.gif', # output gif
                frames,          # array of input frames
                duration=100)         # optional: frames per second
        
        print('GIFF')
        
        #### PREDICT ##########################################################
        model.predict()
        # model.test()
        # model.inference()
        # model.plot_predict()
        print('The end...')
        sys.exit(0)

if __name__ == "__main__":
    
    freeze_support()
    warnings.filterwarnings("ignore")
    
    fire.Fire(RunTask)
    
    # main()
    