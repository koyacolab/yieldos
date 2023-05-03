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

import sys
import os 

class Reseter(Callback):
    def __init__(self, ModelCheckpointPath, milestones):
        super().__init__
        # self.milestones = milestones
        self.ckpt_files = []
        self.milestones = milestones
        self.ModelCheckpointPath = ModelCheckpointPath
        # sys.exit(0)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        print('trainer.should_stop:', trainer.should_stop )
        try:
            # get a list of all the files in the parent directory with a .ckpt extension
            self.ckpt_files = [f for f in os.listdir(self.ModelCheckpointPath) if f.endswith('.ckpt')]
        except FileNotFoundError:
            # handle the case where the parent directory doesn't exist
            print("No checkpoint found, maybe it's first start")
            # self.milestones = 10
            
        if trainer.current_epoch >= self.milestones:
            trainer.should_stop = True
            
        if len(self.ckpt_files) > 0:
            self.milestones = int(self.ckpt_files[0].split('-')[0].split('epoch=')[1]) + 10
        print('Reseter:', self.milestones)
        

        
class ActualVsPredictedCallback(Callback):
    def __init__(self, dataloader, filename='actuals_vs_predictions', milestones=[0, 2, 25, 50, 100, 120]):
        super().__init__
        self.milestones = milestones
        self.dataloader = dataloader
        self.filename = filename
        
    # def on_validation_epoch_end(self, trainer, pl_module):
        # if trainer.current_epoch not in self.milestones:
        #     return
        # print('on_validaton_epoch_end')
        # print('ActPred1', type(pl_module.train_dataloader))
        # print("Start predicting!")
        # this will fetch the dataloader from the LM or LDM
        # pl_module.eval()
        # calculate actuals and predictions        
        # self.writer = SummaryWriter(log_dir=trainer.log_dir)
        # y_true = torch.cat([y[0] for x, y in iter(self.dataloader)])
        # y_pred = y_true
        # with torch.no_grad():
        # y_pred = pl_module.predict(self.dataloader)
        # print(len(y_pred), y_pred[0])
        # sys.exit(0)
        # y_pred = trainer.predict(pl_module, self.dataloader)
        # print(trainer.__doc__)
        # fn
        # y_pred = pl_module(self.dataloader)
        
        # raw_predictions = pl_module.predict(self.dataloader, mode="raw", return_x=True)
        
#         print('ActPred2', y_true.device, y_true.shape, y_pred.device)
        
#         # # Calculate SMAPE for the entire dataset
#         # smape = SMAPE()
#         # smape_val = smape(torch.flatten(y_pred), torch.flatten(y_true))

#         # Create plot
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(y_true.cpu().numpy(), 'o', color='green', label='actuals')
#         ax.plot(y_pred.cpu().numpy(), '.', color='red', label='predictions')
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Value')
#         # ax.set_title(f'Validation SMAPE: {smape_val:.2%}')
#         ax.legend()
        
#         print('ActPred3', y_true.device, y_pred.device)
        
#         # save the plot as an image
#         plt.savefig(f"{self.filename}.png")
        # fn
        
        # log the image to TensorBoard
        # logger = trainer.logger.experiment
        # logger.add_figure('Validation/Actuals vs. Predictions', fig)
        # fn

# #         # Log plot to TensorBoard
# #         # self.writer.add_figure('Validation/Actuals vs. Predictions', fig, global_step=trainer.global_step)

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

class FineTuneLearningRateFinder_CyclicLR2(LearningRateFinder):
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
        lr = trainer.optimizers[0].param_groups[0]['lr']
        print('CycicLR lr:', lr, trainer.current_epoch)
        # self.base_lr = lr
        print("CycicLR:", self.base_lr, self.max_lr, self.step_size_up, self.step_size_down, self.mode)
        self.optimizer = trainer.optimizers[0]
        # lr = trainer.optimizers[0].param_groups[0]['lr']
        # print('lr:', lr)
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
        for ii in range(trainer.current_epoch):
            self.scheduler.step()
        print('on_fit_start:', self.scheduler.get_last_lr()[0])
        print('on_fit_start:', trainer.optimizers[0].param_groups[0]['lr'])
        return

    def on_train_epoch_start(self, trainer, pl_module):
        self.optimizer = trainer.optimizers[0]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.scheduler.get_last_lr()[0]
        self.scheduler.step()
        print('on_train_epoch_start:', self.scheduler.get_last_lr()[0])
        print('on_train_epoch_start:', trainer.optimizers[0].param_groups[0]['lr'])
        
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

