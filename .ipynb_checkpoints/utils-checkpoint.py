import torch 
from pytorch_lightning.callbacks import LearningRateFinder
from pytorch_lightning.callbacks import Callback
from pytorch_forecasting.data import TimeSeriesDataSet



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

# class ReloadDataLoader(Callback):
#     def __init__(self, train_dataset: TimeSeriesDataSet):
#         self.train_dataset = train_dataset
    
#     def on_train_epoch_start(self, trainer, pl_module):
#         pl_module.train_dataloader = self.train_dataset.to_dataloader(batch_size=trainer.batch_size, shuffle=True)
#         print('DataLoader was realoaded...')