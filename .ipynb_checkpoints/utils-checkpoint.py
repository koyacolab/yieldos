import torch 
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import Callback
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting.metrics import MAPE, SMAPE

import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

import random

# -----------------------------------------------------------------------------

from torch.utils.tensorboard import SummaryWriter

# class ActualVsPredictedCallback(Callback):
#     def __init__(self, val_dataloader, milestones=[1, 25, 50, 100, 120]):
#         self.milestones = milestones
#         self.val_dataloader = val_dataloader

#     def on_validation_epoch_end(self, trainer, pl_module):
#         if trainer.current_epoch not in self.milestones:
#             return
#         writer = SummaryWriter(trainer.logger.log_dir)
#         device = pl_module.device

#         # Get the actual and predicted values for the validation set
#         val_loader = self.val_dataloader
#         val_preds = []
#         val_targets = []
#         with torch.no_grad():
#             for batch in val_loader:
#                 x, y = batch
#                 y_pred = pl_module(x.to(device)).cpu().numpy()
#                 val_preds.append(y_pred)
#                 val_targets.append(y.cpu().numpy())
#         val_preds = np.concatenate(val_preds)
#         val_targets = np.concatenate(val_targets)

#         # Create a scatter plot of the actual vs. predicted values
#         fig, ax = plt.subplots()
#         ax.scatter(val_targets, val_preds)
#         ax.set_xlabel("Actual")
#         ax.set_ylabel("Predicted")
#         ax.set_title("Actual vs. Predicted")
#         writer.add_figure("actual_vs_predicted", fig, global_step=trainer.global_step)
#         writer.close()
        
class ActualVsPredictedCallback(Callback):
    def __init__(self, dataloader, filename='actuals_vs_predictions', milestones=[0, 25, 50, 100, 120]):
        self.milestones = milestones
        self.dataloader = dataloader
        self.filename = filename
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch not in self.milestones:
            return
        # calculate actuals and predictions        
        # self.writer = SummaryWriter(log_dir=trainer.log_dir)
        y_true = torch.cat([y[0] for x, y in iter(self.dataloader)])
        y_pred = pl_module.predict(self.dataloader)
        
        # # Calculate SMAPE for the entire dataset
        # smape = SMAPE()
        # smape_val = smape(torch.flatten(y_pred), torch.flatten(y_true))

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_true.cpu().numpy(), 'o', color='green', label='actuals')
        ax.plot(y_pred.cpu().numpy(), '.', color='red', label='predictions')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        # ax.set_title(f'Validation SMAPE: {smape_val:.2%}')
        ax.legend()
        
        # save the plot as an image
        plt.savefig(f"{self.filename}.png")
        
        # log the image to TensorBoard
        logger = trainer.logger.experiment
        logger.add_figure('Validation/Actuals vs. Predictions', fig)

        # Log plot to TensorBoard
        # self.writer.add_figure('Validation/Actuals vs. Predictions', fig, global_step=trainer.global_step)

# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------------------------------------------                  

class FineTuneLearningRateFinder_CustomLR(LearningRateFinder):
    def __init__(self, total_const_iters=20, base_lr=0.001, max_lr=0.085, step_size_up=30, step_size_down=70, mode='triangular2', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.total_const_iters = total_const_iters
        self.mode = mode
        self.optimizer = []
        self.scheduler = []

    def on_fit_start(self, trainer, pl_module):
        print("CustomLR:", self.base_lr, self.max_lr, self.step_size_up, self.step_size_down, self.mode)
        self.optimizer = trainer.optimizers[0]
        self.scheduler.append(torch.optim.lr_scheduler.ConstantLR(self.optimizer, 
                                                             factor=1.0, 
                                                             total_iters=self.total_const_iters, 
                                                             last_epoch=-1, 
                                                             verbose=False))
        
        self.scheduler.append(torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
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
                                                           last_epoch=-1, 
                                                           verbose=False))
        
        # self.scheduler.append(torch.optim.lr_scheduler.ConstantLR(self.optimizer, 
        #                                                      factor=1.0, 
        #                                                      total_iters=self.total_const_iters, 
        #                                                      last_epoch=-1, 
        #                                                      verbose=False))
        
        self.scheduler.append(torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                              step_size=10, 
                                                              gamma=0.1, 
                                                              last_epoch=- 1, 
                                                              verbose=False))
    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler[0].get_last_lr()[0]
        # self.scheduler.step()
        print('on_fit_start:', self.scheduler[0].get_last_lr()[0])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        if trainer.current_epoch <= self.total_const_iters:        
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler[0].get_last_lr()[0]
            self.scheduler[0].step()
            print('on_train_epoch_start:', self.scheduler[0].get_last_lr()[0])
        elif (trainer.current_epoch > self.total_const_iters) & \
             (trainer.current_epoch < self.total_const_iters + self.step_size_up + self.step_size_down):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler[1].get_last_lr()[0]
            self.scheduler[1].step()
            print('on_train_epoch_start:', self.scheduler[1].get_last_lr()[0])
        else:        
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.scheduler[2].get_last_lr()[0]
            self.scheduler[2].step()
            print('on_train_epoch_start:', self.scheduler[2].get_last_lr()[0])
        
# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------                  

class FineTuneLearningRateFinder_StepLR(LearningRateFinder):
    def __init__(self, step_size=50, gamma=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_size = step_size
        self.gamma = gamma
        self.optimizer = []
        self.scheduler = []

    def on_fit_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                         step_size=self.step_size, 
                                                         gamma=self.gamma, 
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
    def __init__(self, data_train, dataset_train, batch_size, YEARS_MAX_LENGTH=1, NSAMPLES=1):
        super().__init__()
        self.data_train = data_train
        # self.data_valid = data_valid
        self.dataset_train = dataset_train
        self.batch_size = batch_size
        self.YEARS_MAX_LENGTH = YEARS_MAX_LENGTH
        self.NSAMPLES = NSAMPLES
        
    def on_train_epoch_start(self, trainer, pl_module):
        # if trainer.current_epoch in self.milestones:
        print('DataGenerator reloading... epoch:', trainer.current_epoch)  
        data_train, year_list = DataGenerator(DATA=self.data_train, 
                                              YEARS_MAX_LENGTH=self.YEARS_MAX_LENGTH, 
                                              NSAMPLES=self.NSAMPLES)
        self.dataset_train = TimeSeriesDataSet.from_dataset(self.dataset_train, data_train)
        pl_module.train_dataloader = self.dataset_train.to_dataloader(batch_size=self.batch_size, shuffle=True)
        print('DataLoader was reloaded...')

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
class CheckpointCallback(Callback):
    def __init__(self, checkpoint_dir, checkpoint_filename, checkpoint_interval):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.checkpoint_interval = checkpoint_interval

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.checkpoint_interval == 0:
            checkpoint_path = os.path.join(self.checkpoint_dir, 
                                           self.checkpoint_filename.format(epoch=epoch+1,
                                           val_loss=trainer.callback_metrics['val_loss']))
            trainer.save_checkpoint(checkpoint_path)

# example 
# trainer = pl.Trainer(callbacks=[CheckpointCallback(checkpoint_dir='checkpoints/', checkpoint_filename='model-{epoch:02d}-{val_loss:.2f}.ckpt', checkpoint_interval=5)])

################# DEPRICIATED #####################################################################
###################################################################################################

def DataGenerator_experimental(TRAIN_DATA, VALID_DATA, YEARS_MAX_LENGTH, ADD_NSAMPLES_LIST=[]):
    years_list = list(TRAIN_DATA['year'].astype(int).unique())
    print(f'Augmentation for train years list: {years_list}')

    valid_years_list = list(VALID_DATA['year'].astype(int).unique())
    print(f'Augmentation for valid years list: {valid_years_list}')
    
    NSAMPLES_LIST = []
    NSAMPLES_LIST.extend(valid_years_list)
    NSAMPLES_LIST.extend(ADD_NSAMPLES_LIST)
    print(f'Augmentation for nsamples list: {NSAMPLES_LIST}')

    # random_years = random.sample(years, LENGTH)

    # start_year = DATA['year'].astype(int).min()
    # end_year = DATA['year'].astype(int).max()

    data_samples = pd.DataFrame()
    years_samples = []
    # for ii in tqdm(range(NSAMPLES)):
    for ii in tqdm(NSAMPLES_LIST):
        # num_years = random.randint(1, YEARS_MAX_LENGTH)  # generate a random number between 1 and 10 for the list size
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = random.sample(years_list, num_years)
        # print('DataGenerator nsamples:', ii, type(years), years)
        # df_concat = pd.DataFrame()
        for county in TRAIN_DATA["county"].unique():
            num_years = random.randint(1, YEARS_MAX_LENGTH)
            years = random.sample(years_list, num_years)
            years_samples.append(years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, TRAIN_DATA.loc[ (TRAIN_DATA['year'].astype(int) == iyear) & \
                                                         (TRAIN_DATA['county'] == county)] ], axis=0)
            # val_year = random.sample(valid_years_list, 1)
            # # print('val_year', val_year[0])
            # # add validation year as a last sample for train/validation splitting
            # df_concat_year = pd.concat([df_concat_year, VALID_DATA.loc[ (VALID_DATA['year'].astype(int) == val_year[0]) & \
            #                                              (VALID_DATA['county'] == county)] ], axis=0)
            # years_samples.append(val_year)
            # reindex the concatenated dataframe with a new index
            new_index = pd.RangeIndex(start=0, stop=len(df_concat_year)+0, step=1)
            df_concat_year.index = new_index
            # add a new column with integer values equal to the index
            df_concat_year["time_idx"] = df_concat_year.index.astype(int)
            df_concat_year["sample"] = str(ii)
            data_samples = pd.concat([data_samples, df_concat_year], axis=0)
        # reindex the concatenated dataframe with a new index
    new_index = pd.RangeIndex(start=1, stop=len(data_samples)+1, step=1)
    data_samples.index = new_index

    return data_samples, years_samples

#################################################################################################

def DataGenerator_split(TRAIN_DATA, VALID_DATA, YEARS_MAX_LENGTH, ADD_NSAMPLES_LIST=[]):
    years_list = list(TRAIN_DATA['year'].astype(int).unique())
    print(f'Augmentation for train years list: {years_list}')

    valid_years_list = list(VALID_DATA['year'].astype(int).unique())
    print(f'Augmentation for valid years list: {valid_years_list}')
    
    NSAMPLES_LIST = []
    NSAMPLES_LIST.extend(valid_years_list)
    NSAMPLES_LIST.extend(ADD_NSAMPLES_LIST)
    print(f'Augmentation for nsamples list: {NSAMPLES_LIST}')

    # random_years = random.sample(years, LENGTH)

    # start_year = DATA['year'].astype(int).min()
    # end_year = DATA['year'].astype(int).max()

    data_samples = pd.DataFrame()
    years_samples = []
    for ii in tqdm(range(len(NSAMPLES_LIST))):
    # for ii in tqdm(NSAMPLES_LIST):
        # num_years = random.randint(1, YEARS_MAX_LENGTH)  # generate a random number between 1 and 10 for the list size
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = [random.randint(start_year, end_year) for _ in range(num_years)]
        # years = random.sample(years_list, num_years)
        # print('DataGenerator nsamples:', ii, type(years), years)
        # df_concat = pd.DataFrame()
        for county in TRAIN_DATA["county"].unique():
            num_years = random.randint(1, YEARS_MAX_LENGTH)
            years = random.sample(years_list, num_years)
            years_samples.append(years)
            df_concat_year = pd.DataFrame()
            for iyear in years:
                df_concat_year = pd.concat([ df_concat_year, TRAIN_DATA.loc[ (TRAIN_DATA['year'].astype(int) == iyear) & \
                                                         (TRAIN_DATA['county'] == county)] ], axis=0)
            val_year = random.sample(valid_years_list, 1)
            # print('val_year', val_year[0])
            # add validation year as a last sample for train/validation splitting
            df_concat_year = pd.concat([df_concat_year, VALID_DATA.loc[ (VALID_DATA['year'].astype(int) == val_year[0]) & \
                                                         (VALID_DATA['county'] == county)] ], axis=0)
            years_samples.append(val_year)
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


# class ReloadDataLoader(Callback):
#     def __init__(self, train_dataset: TimeSeriesDataSet):
#         self.train_dataset = train_dataset
    
#     def on_train_epoch_start(self, trainer, pl_module):
#         pl_module.train_dataloader = self.train_dataset.to_dataloader(batch_size=trainer.batch_size, shuffle=True)
#         print('DataLoader was realoaded...')