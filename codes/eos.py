from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from models import MLPNet, ResMLPNet, CNNNet, ResCNNNet, ResNet18, UNet 
import torch
from utils import get_transform, get_hessian_eigenvalues, plot_acc, plot_loss, plot_sharpness, FastTensorDataset, rk_advance_time
from sklearn.metrics import accuracy_score
import numpy as np
import os
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters   
import matplotlib.pyplot as plt
import wandb
import pandas as pd
import time

class EOSExperiment(pl.LightningModule):
    def __init__(self, config, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.train_dataset, self.val_dataset, self.test_dataset = self.build_dataset()  
        if self.tensor_dataset is None:
            self.tensor_dataset = self.build_tensor_dataset()
        self.tensor_dataloader = torch.utils.data.DataLoader(self.tensor_dataset, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=0)
        self.build_loss()
        self.build_metrics()
        self.build_model()


    def build_dataset(self):
        self.tensor_dataset = None
        self.train_index = None
        if self.config.dataset.enable_aug:
            return self.build_dataset_wt_aug()
        else:
            return self.build_dataset_wo_aug()
    

    def build_dataset_wt_aug(self):
        train_transform = get_transform(self.config.dataset.train_transform)
        test_transform = get_transform(self.config.dataset.test_transform)
        if self.config.dataset.dataset_type == "cifar10":
            train_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=True, download=True, transform=train_transform)
            test_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=False, download=True, transform=test_transform)
            self.config.model.input_dim = 32*32*3
            self.config.model.spatial_dim = 32
            self.config.model.num_colors = 3
            self.config.model.output_dim = 10
        elif self.config.dataset.dataset_type == "mnist":
            train_dataset = datasets.MNIST(root=self.config.dataset.root, train=True, download=True, transform=train_transform)
            test_dataset = datasets.MNIST(root=self.config.dataset.root, train=False, download=True, transform=test_transform)
            self.config.model.input_dim = 28*28
            self.config.model.spatial_dim = 28
            self.config.model.num_colors = 1
            self.config.model.output_dim = 10
        if self.config.model.loss_type == 'mse' or self.config.model.loss_type == 'mae':
            # convert labels data in dataset to one-hot
            train_dataset.targets = F.one_hot(torch.tensor(train_dataset.targets), num_classes=self.config.model.output_dim).numpy()
            test_dataset.targets = F.one_hot(torch.tensor(test_dataset.targets), num_classes=self.config.model.output_dim).numpy()

        if self.config.dataset.val_ratio > 0:
            val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [int(len(test_dataset)*(1-self.config.dataset.val_ratio)), int(len(test_dataset)*self.config.dataset.val_ratio)])
        else:
            val_dataset = None

        if self.config.dataset.train_num > 0:
            self.train_index = np.random.choice(len(train_dataset), self.config.dataset.train_num, replace=False).reshape(-1)
            train_dataset = torch.utils.data.Subset(train_dataset, self.train_index)

        if self.config.optim.optimizer_type == 'gd':
            # use all data for training
            self.config.dataset.batch_size = len(train_dataset)
        if self.config.dataset.test_num > 0:
            self.test_index = np.random.choice(len(test_dataset), self.config.dataset.test_num, replace=False).reshape(-1)
            test_dataset = torch.utils.data.Subset(test_dataset, self.test_index)
        return train_dataset, val_dataset, test_dataset
    


    def build_dataset_wo_aug(self):
        # Used for Hessians computation
         # reinit the dataset, since there is no transform
        if self.config.dataset.dataset_type == "cifar10":
            train_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=True, download=True)
            test_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=False, download=True)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            max_p = 255
        
        elif self.config.dataset.dataset_type == "mnist":
            train_dataset = datasets.MNIST(root=self.config.dataset.root, train=True, download=True)
            test_dataset = datasets.MNIST(root=self.config.dataset.root, train=False, download=True)
            mean = (0.1307,)
            std = (0.3081,)
            max_p = 255
        #print('??????? ', self.config.dataset.train_num )
        if self.train_index is None:
            if self.config.dataset.train_num > 0:
                self.train_index = np.random.choice(len(train_dataset), self.config.dataset.train_num, replace=False).reshape(-1)
            else:
                self.train_index = np.arange(len(train_dataset))

            if self.config.dataset.test_num > 0:
                self.test_index = np.random.choice(len(test_dataset), self.config.dataset.test_num, replace=False).reshape(-1)
            else:
                self.test_index = np.arange(len(test_dataset))
    
        train_data = np.asarray(train_dataset.data)[self.train_index, ...] / max_p
        if self.config.task == 'denoise':
            train_labels = train_data.copy()
            # add noise to the data
            train_data = train_data + self.config.dataset.noise_level * np.random.randn(*train_data.shape)
        else:
            train_labels = np.asarray(train_dataset.targets)[self.train_index, ...]
        # according to loss type, reshape the label
        if (self.config.model.loss_type == 'mse' or self.config.model.loss_type == 'mae') and self.config.task != 'denoise':
            # convert labels data in dataset to one-hot
            train_labels = F.one_hot(torch.tensor(train_labels), num_classes=self.config.model.output_dim).numpy()
        # normalize the data
        train_data = (train_data - mean) / std
        # permute the data
        train_data = np.transpose(train_data, axes=(0, 3, 1, 2))
        if self.config.task == 'denoise':
            train_labels = (train_labels - mean) / std
            train_labels = np.transpose(train_labels, axes=(0, 3, 1, 2))
        
        train_dataset = FastTensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
        #print(len(train_dataset))
        test_data = np.asarray(test_dataset.data)[self.test_index, ...] / max_p
        test_labels = np.asarray(test_dataset.targets)[self.test_index, ...]

  
        # according to loss type, reshape the label
        if self.config.model.loss_type == 'mse' or self.config.model.loss_type == 'mae':
            # convert labels data in dataset to one-hot
            test_labels = F.one_hot(torch.tensor(test_labels), num_classes=self.config.model.output_dim).numpy()
        # normalize the data
        test_data = (test_data - mean) / std
        # permute the data
        test_data = np.transpose(test_data, axes=(0, 3, 1, 2))
        test_dataset = FastTensorDataset(torch.tensor(test_data).float(), torch.tensor(test_labels).float())
        if self.config.dataset.val_ratio > 0:
            val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [int(len(test_dataset)*(1-self.config.dataset.val_ratio)), int(len(test_dataset)*self.config.dataset.val_ratio)])
        else:
            val_dataset = None
        self.tensor_dataset = train_dataset
        
        if self.config.optim.optimizer_type == 'gd':
            # use all data for training
            self.config.dataset.batch_size = len(train_dataset)
        return train_dataset, val_dataset, test_dataset
    

    def build_tensor_dataset(self):
        if self.config.dataset.dataset_type == "cifar10":
            train_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=True, download=True)
            test_dataset = datasets.CIFAR10(root=self.config.dataset.root, train=False, download=True)
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
            max_p = 255
        
        elif self.config.dataset.dataset_type == "mnist":
            train_dataset = datasets.MNIST(root=self.config.dataset.root, train=True, download=True)
            test_dataset = datasets.MNIST(root=self.config.dataset.root, train=False, download=True)
            mean = (0.1307,)
            std = (0.3081,)
            max_p = 255
        
        if self.train_index is None:
            if self.config.dataset.train_num > 0:
                self.train_index = np.random.choice(len(train_dataset), self.config.dataset.train_num, replace=False).reshape(-1)
            else:
                self.train_index = np.arange(len(train_dataset))

            if self.config.dataset.test_num > 0:
                self.test_index = np.random.choice(len(test_dataset), self.config.dataset.test_num, replace=False).reshape(-1)
            else:
                self.test_index = np.arange(len(test_dataset))
    

        train_data = np.asarray(train_dataset.data)[self.train_index, ...] / max_p
        train_data = np.transpose(train_data, axes=(0, 3, 1, 2))
        train_labels = np.asarray(train_dataset.targets)[self.train_index, ...]


        # according to loss type, reshape the label
        if self.config.model.loss_type == 'mse' or self.config.model.loss_type == 'mae':
            # convert labels data in dataset to one-hot
            train_labels = F.one_hot(torch.tensor(train_labels), num_classes=self.config.model.output_dim).numpy()
        # normalize the data [5000, 3, 32, 32] by mean and std [3, ]
        mean = np.expand_dims(np.expand_dims(np.expand_dims(mean, axis=0), axis=2), axis=3)
        std = np.expand_dims(np.expand_dims(np.expand_dims(std, axis=0), axis=2), axis=3)
        train_data = (train_data - mean) / std
        
        train_dataset = FastTensorDataset(torch.tensor(train_data).float(), torch.tensor(train_labels).float())
        if self.config.optim.optimizer_type == 'gd':
            # use all data for training
            self.config.dataset.batch_size = len(train_dataset)
        return train_dataset


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config.dataset.batch_size, shuffle=True, num_workers=self.config.dataset.num_workers)
    
    def val_dataloader(self):
        if self.val_dataset is not None:
            return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=self.config.dataset.num_workers)
        return None
    
    def test_dataloader(self):
        if self.test_dataset is not None:
            return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config.dataset.batch_size, shuffle=False, num_workers=self.config.dataset.num_workers)
        return None
    
    def build_model(self):
        if self.config.model.model_type == 'mlp':
            self.model = MLPNet(self.config.model.hidden_unit_list, input_dim=self.config.model.input_dim, output_dim=self.config.model.output_dim, activation=self.config.model.activation, norm=self.config.model.norm)
            
        elif self.config.model.model_type == 'resmlp':
            self.model = ResMLPNet(self.config.model.hidden_unit_list, input_dim=self.config.model.input_dim, output_dim=self.config.model.output_dim, activation=self.config.model.activation, norm=self.config.model.norm)
        elif self.config.model.model_type == 'cnn':
            self.model = CNNNet(self.config.model.hidden_unit_list,  output_dim=self.config.model.output_dim, activation=self.config.model.activation, norm=self.config.model.norm, num_colors=self.config.model.num_colors, spatial_dim=self.config.model.spatial_dim)
        elif self.config.model.model_type == 'rescnn':
            self.model = ResCNNNet(self.config.model.hidden_unit_list,  output_dim=self.config.model.output_dim, activation=self.config.model.activation, norm=self.config.model.norm, num_colors=self.config.model.num_colors, spatial_dim=self.config.model.spatial_dim)
        elif self.config.model.model_type == 'resnet18':
            if self.config.model.load_from == 'imagenet':
                pretrained = True
            else:
                pretrained = False
            self.model = ResNet18(output_dim=self.config.model.output_dim, pretrained=pretrained)
            if self.config.model.load_from is not None:
                new_dict = {}
                old_dict = torch.load(self.config.model.load_from, map_location='cpu')
                for key, value in old_dict.items():
                    if 'model.predictor' not in key:
                        new_dict[key] = value
                self.model.load_state_dict(new_dict, strict=False)
        
        elif self.config.model.model_type == 'unet':
            self.model = UNet()
        else:
            raise NotImplementedError
        
    def build_loss(self):
        if self.config.model.loss_type == 'cross_entropy':
            self.loss = lambda y_hat, y: F.cross_entropy(y_hat, y.long())
        elif self.config.model.loss_type == 'mse':
            self.loss = lambda y_hat, y: F.mse_loss(y_hat, y.float(), reduction='none').sum(dim=1).mean()
        elif self.config.model.loss_type == 'mae':
            self.loss = lambda y_hat, y: F.l1_loss(y_hat, y.float(), reduction='none').sum(dim=1).mean()
        else:
            raise NotImplementedError

    def build_metrics(self):
        if self.config.task != 'denoise':
            if self.config.model.loss_type == 'cross_entropy':
                self.metrics = lambda y_hat, y: accuracy_score(y.argmax(axis=1), y_hat)
            elif self.config.model.loss_type == 'mse' or self.config.model.loss_type == 'mae':
                self.metrics = lambda y_hat, y: accuracy_score(y.argmax(axis=1), y_hat.argmax(axis=1))
        else:
            # PSNR
            self.metrics = lambda y_hat, y: (10 * np.log10(1 / np.mean((y_hat - y)**2))).mean()
            
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.config.optim.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay, betas=(self.config.optim.beta1, self.config.optim.beta2))
        elif self.config.optim.optimizer_type == "sgd" or self.config.optim.optimizer_type == "gd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer_type == 'polyak':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.optim.lr, momentum=self.config.optim.momentum, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer_type == 'nesterov':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.optim.lr, momentum=self.config.optim.momentum, nesterov=True, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer_type == 'flow':
            # set manual optimization in lightning
            self.automatic_optimization = False
            return None
        if self.config.optim.scheduler_type is None:
            return [optimizer]
        if self.config.optim.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.optim.step_size, gamma=self.config.optim.gamma)
        elif self.config.optim.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.optim.T_max, eta_min=self.config.optim.eta_min)
        else:
            raise NotImplementedError
        
        return [optimizer], [scheduler]
    

    def on_train_start(self):
        self.train_loss_history = []
        self.train_acc_history = []
        self.test_loss_history = []
        self.test_acc_history = []
        self.eign_history = []
        if self.config.nproj > 0:
            self.projectors = torch.randn(self.config.nproj, len(parameters_to_vector(self.parameters())))
        else:
            self.projectors = None
        self.iteration_list = []
        self.last_sharpness = get_hessian_eigenvalues(self.model, self.loss, self.tensor_dataset, neigs=self.config.n_eigen, batch_size=self.config.dataset.batch_size, device=self.device)[0]
        if self.config.optim.sharpness_schedule is not None:
            self.optimizers().param_groups[0]['lr'] = 1 / self.last_sharpness
        if self.config.dataset.fast_load:
            if not self.config.dataset.enable_aug:
                #print(self.train_dataset.tensors)
                self.train_dataset.to(self.device)
                self.test_dataset.to(self.device)
            self.tensor_dataset.to(self.device)
    
    def shared_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        # check loss is NAN or not / is INF or not
        if torch.isnan(loss) or torch.isinf(loss):
            # stop training
            self.trainer.should_stop = True
        #self.log('train_step_loss', loss, prog_bar=True)
        print(f'self.current_step: {self.trainer.global_step} | loss: {loss} | last_sharpness: {self.last_sharpness}')
        return loss
    
    def compute_all_on_whole_dataset(self):
        # compute all again on whole train and test dataset
        # train
        train_loss = 0
        train_labels = []
        train_preds = []
        for batch in self.tensor_dataloader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            train_loss += loss.item()
            train_labels.append(y.detach().cpu().numpy())
            train_preds.append(y_hat.detach().cpu().numpy())

        #print(len(self.train_dataset))
        #print(self.config.dataset.batch_size)
        train_loss = train_loss / (len(self.train_dataset) / self.config.dataset.batch_size)
        train_labels = np.concatenate(train_labels, axis=0)
        train_preds = np.concatenate(train_preds, axis=0)


        # log train loss & train_acc
        #self.log('train_loss', train_loss, prog_bar=True, on_epoch=True)
        print(f'Train Epoch: {self.current_epoch} | loss: {train_loss}')
        train_acc = self.metrics(train_labels, train_preds)
        #self.log('train_acc', train_acc, prog_bar=True, on_epoch=True)
        print(f'Train Epoch: {self.current_epoch} | acc: {train_acc}')
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        
        if self.config.no_test:
            return
        test_loss = 0
        test_labels = []
        test_preds = []
        for batch in self.test_dataloader():
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self(x)
            loss = self.loss(y_hat, y)
            test_loss += loss.item()
            test_labels.append(y.detach().cpu().numpy())
            test_preds.append(y_hat.detach().cpu().numpy())
        test_loss =  test_loss / (len(self.test_dataset)  / self.config.dataset.batch_size)
        
        test_labels = np.concatenate(test_labels, axis=0)
        test_preds = np.concatenate(test_preds, axis=0)
        # log test loss & test_acc
        #self.log('test_loss', test_loss, prog_bar=True, on_epoch=True)
        test_acc = self.metrics(test_labels, test_preds)

        #self.log('test_acc', test_acc, prog_bar=True, on_epoch=True)
        print(f'Test Epoch: {self.current_epoch} | loss: {test_loss}')
        print(f'Test Epoch: {self.current_epoch} | acc: {test_acc}')
        # save to history

        self.test_loss_history.append(test_loss)
        self.test_acc_history.append(test_acc)


    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log('val_step_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        pass # do nothing 

    def on_train_epoch_start(self) -> None:
        if self.config.eigen_freq > 0 and (((self.current_epoch - 1) % self.config.eigen_freq == 0) or (self.current_epoch == 0)) and self.config.optim.sharpness_schedule == 'dynamic':
            # set learning rate in optimizer to be 1 / sharpness
            self.optimizers().param_groups[0]['lr'] = 1 / self.last_sharpness
            print(f'Epoch: {self.current_epoch} | lr: {self.optimizers().param_groups[0]["lr"]}')
            
            
    def on_train_epoch_end(self) -> None:
        if not self.automatic_optimization:
            rk_step_size = min(self.config.optim.flow_alpha / self.last_sharpness, self.config.optim.max_step_size)
            rk_advance_time(self.model, self.loss, self.tensor_dataset, self.config.optim.tick, rk_step_size, self.config.dataset.batch_size, self.device)
        torch.set_grad_enabled(False)
        self.model.eval()
        self.compute_all_on_whole_dataset()
        torch.set_grad_enabled(True)
        self.model.train()
        if self.config.eigen_freq > 0 and self.current_epoch % self.config.eigen_freq == 0:
            eigenvalues = get_hessian_eigenvalues(self.model, self.loss, self.tensor_dataset, neigs=self.config.n_eigen, batch_size=self.config.dataset.batch_size, device=self.device)
            # replace NAN with 2^24
            #eigenvalues = np.nan_to_num(eigenvalues, nan=2^24)
            # clamp eigenvalues
            #eigenvalues = np.clip(eigenvalues, a_min=-2^24, a_max=2^24)
            #self.log('eigenvalues', eigenvalues[0], prog_bar=True, on_epoch=True)
            print(f'Epoch: {self.current_epoch} | eigenvalues: {eigenvalues[0]}')
            self.last_sharpness = eigenvalues[0]
            self.eign_history.append(eigenvalues)
        if self.config.iterate_freq  > 0 and self.current_epoch % self.config.iterate_freq == 0:
            self.iteration_list.append(self.projectors.mv(parameters_to_vector(self.parameters()).detach().cpu()).numpy())
        

    def on_train_end(self) -> None:
        # replace NAN with 2^24
        self.eign_history = np.nan_to_num(self.eign_history, nan=2^24)
        self.train_loss_history = np.nan_to_num(self.train_loss_history, nan=2^24)
        self.train_loss_history = np.clip(self.train_loss_history, a_min=0, a_max=2^24)



        # save train and test loss and acc history
        os.makedirs(self.config.hist_save_dir, exist_ok=True)
        if self.config.iterate_freq > 0:
            np.save(os.path.join(self.config.hist_save_dir, 'iteration.npy'), np.array(self.iteration_list))
        if self.config.nproj > 0:
            np.save(os.path.join(self.config.hist_save_dir, 'projectors.npy'), self.projectors.numpy())
        if self.config.eigen_freq > 0:
            self.eign_history = np.array(self.eign_history)
            np.save(os.path.join(self.config.hist_save_dir, 'eigenvalues.npy'), self.eign_history )
            self.sharpness = self.eign_history[:, 0]
        # plot all
        np.save(os.path.join(self.config.hist_save_dir, 'train_loss.npy'), np.array(self.train_loss_history))
        np.save(os.path.join(self.config.hist_save_dir, 'train_acc.npy'), np.array(self.train_acc_history))

        if not self.config.no_test:
            self.test_loss_history = np.nan_to_num(self.test_loss_history, nan=2^24)
            self.test_loss_history = np.clip(self.test_loss_history, a_min=0, a_max=2^24)
            np.save(os.path.join(self.config.hist_save_dir, 'test_loss.npy'), np.array(self.test_loss_history))
            np.save(os.path.join(self.config.hist_save_dir, 'test_acc.npy'), np.array(self.test_acc_history))
       
        self.plot_all()
        self.upload_curves()
        
    
    def plot_all(self):
        plot_acc(self.train_acc_history, title='Train Acc', save_path=os.path.join(self.config.hist_save_dir, 'train_acc.png'))
        # upload to wandb
        self.logger.experiment[0].log({'train_acc_plot': plt})
        plt.close()
       
        plot_loss(self.train_loss_history, title='Train Loss', save_path=os.path.join(self.config.hist_save_dir, 'train_loss.png'))
        self.logger.experiment[0].log({'train_loss_plot': plt})
        plt.close()


        if not self.config.no_test:
            plot_acc(self.test_acc_history, title='Test Acc', save_path=os.path.join(self.config.hist_save_dir, 'test_acc.png'))
            self.logger.experiment[0].log({'test_acc_plot': plt})
            plt.close()
            plot_loss(self.test_loss_history, title='Test Loss', save_path=os.path.join(self.config.hist_save_dir, 'test_loss.png'))
            self.logger.experiment[0].log({'test_loss_plot': plt})
            plt.close()
        if self.config.eigen_freq > 0:
            if self.config.optim.optimizer_type == 'gd' or self.config.optim.optimizer_type == 'sgd':
                gd_lr_line = 2 / self.config.optim.lr
            elif self.config.optim.optimizer_type == 'polyak':
                gd_lr_line = (2 + 2 * self.config.optim.momentum) / self.config.optim.lr
            elif self.config.optim.optimizer_type == 'nesterov':
                gd_lr_line = (2 + 2 * self.config.optim.momentum) / (1 + self.config.optim.momentum) / self.config.optim.lr
            else:
                gd_lr_line = None
            plot_sharpness(self.sharpness,  save_path=os.path.join(self.config.hist_save_dir, 'sharpness.png'), eign_freq=self.config.eigen_freq, gd_lr_line = gd_lr_line)
            self.logger.experiment[0].log({'sharpness_plot': plt})
            plt.close()
        

        # upload to wandb
        #self.logger.experiment[0].log({'train_acc_plot': train_acc_plot, 'test_acc_plot': test_acc_plot, 'train_loss_plot': train_loss_plot, 'test_loss_plot': test_loss_plot, 'sharpness_plot': sharpness_plot})
    
    def upload_curves(self):
        # upload to wandb
        train_loss_table_matrix = np.concatenate([np.arange(0, len(self.train_loss_history)).reshape(-1, 1), np.array(self.train_loss_history).reshape(-1, 1)], axis=1)
        train_acc_table_matrix = np.concatenate([np.arange(0, len(self.train_acc_history)).reshape(-1, 1), np.array(self.train_acc_history).reshape(-1, 1)], axis=1)
       
        self.logger.experiment[0].log({"train_loss": wandb.plot.line(table=wandb.Table(data=train_loss_table_matrix, columns=["step", "train_loss"]), x="step", y="train_loss", title="Train Loss")})
        self.logger.experiment[0].log({"train_acc": wandb.plot.line(table=wandb.Table(data=train_acc_table_matrix, columns=["step", "train_acc"]), x="step", y="train_acc", title="Train Acc")})
        
        if not self.config.no_test:
            test_loss_table_matrix = np.concatenate([np.arange(0, len(self.test_loss_history)).reshape(-1, 1), np.array(self.test_loss_history).reshape(-1, 1)], axis=1)
            test_acc_table_matrix = np.concatenate([np.arange(0, len(self.test_acc_history)).reshape(-1, 1), np.array(self.test_acc_history).reshape(-1, 1)], axis=1)
            self.logger.experiment[0].log({"test_loss": wandb.plot.line(table=wandb.Table(data=test_loss_table_matrix, columns=["step", "test_loss"]), x="step", y="test_loss", title="Test Loss")})
            self.logger.experiment[0].log({"test_acc": wandb.plot.line(table=wandb.Table(data=test_acc_table_matrix, columns=["step", "test_acc"]), x="step", y="test_acc", title="Test Acc")})
        if self.config.eigen_freq > 0:
            sharpness_table_matrix = np.concatenate([np.arange(0, len(self.sharpness) * self.config.eigen_freq, self.config.eigen_freq).reshape(-1, 1), self.sharpness.reshape(-1, 1)], axis=1)
            self.logger.experiment[0].log({"sharpness": wandb.plot.line(table=wandb.Table(data=sharpness_table_matrix, columns=["step", "sharpness"]), x="step", y="sharpness", title="Sharpness")})




