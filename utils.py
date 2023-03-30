import torch 
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import Callback
from pytorch_forecasting.data import TimeSeriesDataSet

import pandas as pd

from tqdm import tqdm

import random

class FineTuneLearningRateFinder_0(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)   
            
# ---------------------------------------------------------------------------------------------------------------           

class FineTuneLearningRateFinder_1(LearningRateFinder):
    def __init__(self, milestones, gamma=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones
        self.gamma = gamma
        self.optimizer = []
        self.scheduler = []
        # self.optimizer = []
        # self.scheduler = []

    def on_fit_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.5, total_iters=50)
        # StepLR(optimizer, self.step_size, self.gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_fit_start:', self.scheduler.get_last_lr()[0])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.optimizer = trainer.optimizers[0]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
            # StepLR(optimizer, self.step_size, self.gamma)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
        
# ---------------------------------------------------------------------------------------------------------------  
# ---------------------------------------------------------------------------------------------------------------           

class FineTuneLearningRateFinder_LinearLR(LearningRateFinder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = []
        self.scheduler = []
        # self.optimizer = []
        # self.scheduler = []

    def on_fit_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.35, total_iters=70)
        # StepLR(optimizer, self.step_size, self.gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_fit_start:', self.scheduler.get_last_lr()[0])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.35, total_iters=70)
        # StepLR(optimizer, self.step_size, self.gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
        
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------           

# class FineTuneLearningRateFinder_CyclicLR(LearningRateFinder):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.optimizer = []
#         self.scheduler = []
#         # self.optimizer = []
#         # self.scheduler = []

#     def on_fit_start(self, trainer, pl_module):
#         self.optimizer = trainer.optimizers[0]
#         # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
#         self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.35, end_factor=1.0, total_iters=70)
#         # StepLR(optimizer, self.step_size, self.gamma)
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = self.scheduler.get_last_lr()[0]
#         self.scheduler.step()
#         print('on_fit_start:', self.scheduler.get_last_lr()[0])
#         return

#     def on_train_epoch_start(self, trainer, pl_module):
#         self.optimizer = trainer.optimizers[0]
#         if trainer.current_epoch == 0:
#             self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.35, end_factor=1.0, total_iters=4)
#         elif trainer.current_epoch == 4:
#             self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.35, total_iters=4)
#         # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
#         # self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.35, end_factor=1.0, total_iters=70)
#         # StepLR(optimizer, self.step_size, self.gamma)
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = self.scheduler.get_last_lr()[0]
#         self.scheduler.step()
#         print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
        
# ---------------------------------------------------------------------------------------------------------------           

class FineTuneLearningRateFinder_CyclicLR(LearningRateFinder):
    def __init__(self, base_lr=0.001, max_lr=0.085, step_size_up=30, step_size_down=70, mode='triangular2', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.optimizer = []
        self.scheduler = []

    def on_fit_start(self, trainer, pl_module):
        print("CycicLR:", self.base_lr, self.max_lr, self.step_size_up, self.step_size_down, self.mode)
        self.optimizer = trainer.optimizers[0]
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                           base_lr=self.base_lr, 
                                                           max_lr=self.max_lr, 
                                                           step_size_up=self.step_size_up, 
                                                           step_size_down=self.step_size_down, 
                                                           mode=self.mode, 
                                                           gamma=1.0, 
                                                           scale_fn=None, 
                                                           scale_mode='cycle', 
                                                           cycle_momentum=True, 
                                                           base_momentum=0.8, 
                                                           max_momentum=0.9, 
                                                           last_epoch=- 1, 
                                                           verbose=False)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        # self.scheduler.step()
        print('on_fit_start:', self.scheduler.get_last_lr()[0])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
        
# ---------------------------------------------------------------------------------------------------------------

class FineTuneLearningRateFinder_2(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones
        self.gamma = 0.5
        self.optimizer = []
        self.scheduler = []

    # def on_fit_start(self, *args, **kwargs):
    def on_fit_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        print('find initial lr')
        self.lr_find(trainer, pl_module)
        # StepLR(optimizer, self.step_size, self.gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_fit_start:', self.scheduler.get_last_lr()[0])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        # if trainer.current_epoch == 0:       
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.optimizer = trainer.optimizers[0]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.milestones, self.gamma)
        # StepLR(optimizer, self.step_size, self.gamma)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
            
    def on_train_epoch_end(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.optimizer.param_groups[0]['capturable'] = True
    
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
    def __init__(self, data_train, dataset_train, batch_size):
        super().__init__()
        self.data_train = data_train
        self.dataset_train = dataset_train
        self.batch_size = batch_size
        
    def on_train_epoch_start(self, trainer, pl_module):
        # if trainer.current_epoch in self.milestones:
        print('DataGenerator reloading... epoch:', trainer.current_epoch)  
        data_train = DataGenerator(DATA=self.data_train, YEARS_MAX_LENGTH=4, NSAMPLES=4)
        self.dataset_train = TimeSeriesDataSet.from_dataset(self.dataset_train, data_train)
        pl_module.train_dataloader = self.dataset_train.to_dataloader(batch_size=self.batch_size, shuffle=True)
        print('DataLoader was reloaded...')
        
def DataGenerator(DATA, YEARS_MAX_LENGTH, NSAMPLES):
    years_list = list(DATA['year'].astype(int).unique())
    print(f'Augmentation for years list: {years_list}')

    # random_years = random.sample(years, LENGTH)

    start_year = DATA['year'].astype(int).min()
    end_year = DATA['year'].astype(int).max()

    data_samples = pd.DataFrame()
    for ii in tqdm(range(NSAMPLES)):
        # num_years = random.randint(1, YEARS_MAX_LENGTH)  # generate a random number between 1 and 10 for the list size
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = random.sample(years_list, num_years)
        # print('DataGenerator nsamples:', ii, type(years), years)
        # df_concat = pd.DataFrame()
        for county in DATA["county"].unique():
            num_years = random.randint(1, YEARS_MAX_LENGTH)
            years = random.sample(years_list, num_years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, DATA.loc[ (DATA['year'].astype(int) == iyear) & \
                                                         (DATA['county'] == county)] ], axis=0)
            # reindex the concatenated dataframe with a new index
            new_index = pd.RangeIndex(start=1, stop=len(df_concat_year)+1, step=1)
            df_concat_year.index = new_index
            # add a new column with integer values equal to the index
            df_concat_year["time_idx"] = df_concat_year.index.astype(int)
            df_concat_year["sample"] = str(ii)
            data_samples = pd.concat([data_samples, df_concat_year], axis=0)
        # reindex the concatenated dataframe with a new index
    new_index = pd.RangeIndex(start=1, stop=len(data_samples)+1, step=1)
    data_samples.index = new_index

    return data_samples

# class ReloadDataLoader(Callback):
#     def __init__(self, train_dataset: TimeSeriesDataSet):
#         self.train_dataset = train_dataset
    
#     def on_train_epoch_start(self, trainer, pl_module):
#         pl_module.train_dataloader = self.train_dataset.to_dataloader(batch_size=trainer.batch_size, shuffle=True)
#         print('DataLoader was realoaded...')