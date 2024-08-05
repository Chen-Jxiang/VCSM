import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from torch import optim
from torch import Tensor

from typing import Tuple
from torchtuples import TupleTree

import os
import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from monai.networks.nets import FullyConnectedNet, VarFullyConnectedNet
from torch.nn import BCEWithLogitsLoss, Linear, CrossEntropyLoss, Softmax, Dropout, Sequential, ReLU

from lifelines.utils import concordance_index


class CoxModel(torch.nn.Module):    
    def __init__(self, D_dat=[250], encoder_var_layer=[1024], decoder_layer=[32], 
                 seed=0, silent=False, lr=1e-5, 
                 alpha=0., beta=1., gamma1=0., gamma2=0., 
                 N_validate=50, filename="models/model"):
        """
        Initialize the VCSM model.
        
        Args:
            D_dat (list): Dimensions of input data for each view.
            encoder_var_layer (list): Dimensions of encoder variance layers.
            decoder_layer (list): Dimensions of decoder layers.
            seed (int): Random seed for reproducibility.
            silent (bool): If True, suppress output during training.
            lr (float): Learning rate for optimizer.
            alpha, beta, gamma1, gamma2 (float): Hyperparameters for loss function.
            N_validate (int): Number of steps between validations.
            filename (str): Base filename for saving model and logs.
        """

        super(cox_model, self).__init__()        
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Hyperparameters
        self.D_dat = D_dat
        self.n_views = len(D_dat)
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        
        self.filename = filename
        self.N_validate = N_validate
        self.silent = silent
        
        # Initialize encoder layers
        self.encoder_mu = torch.nn.ModuleList([
            Linear(D_dat[i], 1) for i in range(self.n_views)
        ])
        
        self.encoder_var = torch.nn.ModuleList([
            FullyConnectedNet(D_dat[i], 1, encoder_var_layer, act='PRELU', dropout=dropout_encoder[i])
            for i in range(self.n_views)
        ])
        
        # Initialize optimizer
        self.optim = optim.Adam(self.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8)
        
        self.loss_opt = 1e30
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick for Gaussian distribution.
        
        Args:
            mu (Tensor): Mean of the Gaussian distribution.
            log_var (Tensor): Log variance of the Gaussian distribution.
        
        Returns:
            Tensor: Sampled values from the Gaussian distribution.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    
    def KL_Divergence(self, mu, log_var):
        """
        Compute KL divergence between a Gaussian distribution and the standard Gaussian.
        
        Args:
            mu (Tensor): Mean of the Gaussian distribution.
            log_var (Tensor): Log variance of the Gaussian distribution.
        
        Returns:
            Tensor: KL divergence value.
        """
        return 0.5 * (-1 - log_var + log_var.exp() + mu.pow(2))

    def cox_ph_loss(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """
        Compute the Cox Proportional Hazards loss.
        
        Args:
            log_h (Tensor): Log hazard predictions.
            durations (Tensor): Time-to-event or censoring time.
            events (Tensor): Event indicators (1 for event, 0 for censoring).
            eps (float): Small value to avoid numerical instability.
        
        Returns:
            Tensor: Computed loss value.
        """
        assert events.sum() > 0
        
        # Sort by descending duration
        idx = durations.sort(descending=True)[1]
        events = events[idx].view(-1)
        log_h = log_h[idx].view(-1)
        
        log_h_max = log_h.max()
        log_h_cumsum = torch.log(torch.exp(log_h - log_h_max).cumsum(0) + eps) + log_h_max
        
        return -((log_h - log_h_cumsum) * events).sum() / events.sum()

    
    def PoG(self, mu_vnd, log_var_vnd):
        """
        Product of Gaussians: Compute mean and variance of the product of Gaussian distributions.
        
        Args:
            mu_vnd (Tensor): Means of Gaussian distributions.
            log_var_vnd (Tensor): Log variances of Gaussian distributions.
        
        Returns:
            Tuple[Tensor, Tensor]: Mean and log variance of the product distribution.
        """
        log_var_max = log_var_vnd.max()
        var_vnd = torch.exp(log_var_vnd - log_var_max)
        var_nd = 1. / (1. / var_vnd).sum(0)
        mu_nd = var_nd * (1. / var_vnd * mu_vnd).sum(0)
        log_var_nd = torch.log(var_nd) + log_var_max
        
        return mu_nd, log_var_nd

    def forward(self, X, non_prop=False, missing_mask_vn=None):
        """
        Forward pass of the model.
        
        Args:
            X (list): List of input tensors for each view.
            non_prop (bool): If True, use mean instead of reparameterization.
            missing_mask_vn (Tensor): Mask for missing data.
        
        Returns:
            Tuple: Predicted hazards, view-specific hazards, means, and log variances.
        """
        
        N = X[0].shape[0]
        
        mu_vn = torch.zeros( [0, N]  ).cuda()
        logvar_vn = torch.zeros( [0, N]  ).cuda()
        h_pred_vn = torch.zeros( [0, N]  ).cuda()

        
        for v in range(self.n_views):
            X_v = self.dropout(X[v])
            
            mu_v = self.encoder_mu[v](X_v)
            logvar_v = self.encoder_var[v](X_v)
            
            if missing_mask_vn is not None:
                mu_v = torch.einsum("nd, n -> nd", mu_v, missing_mask_vn[v])
                logvar_v = torch.einsum("nd, n -> nd", logvar_v, missing_mask_vn[v]) + \
                           torch.einsum("nd, n -> nd", 5. * torch.ones(logvar_v.shape).cuda(), 1 - missing_mask_vn[v])
            
            mu_vn = torch.cat([mu_vn, mu_v[None, :, 0]])
            logvar_vn = torch.cat([logvar_vn, logvar_v[None, :, 0]])
            
            z_v = mu_v if non_prop else self.reparameterize(mu_v, logvar_v)
            h_pred_v = z_v
            h_pred_vn = torch.cat([h_pred_vn, h_pred_v[None, :, 0]])
        
        mu_n, logvar_n = self.PoG(mu_vn, logvar_vn)
        z_n = mu_n if non_prop else self.reparameterize(mu_n, logvar_n)
        h_pred_n = z_n.flatten()
        
        return h_pred_n, h_pred_vn, mu_vn, logvar_vn
    
    
    def train_model(self, X_train, D_train, E_train, 
                    X_val, D_val, E_val, 
                    X_test=None, D_test=None, E_test=None, 
                    n_step=20000, stop_step=500, min_step=200):
        """
        Train the model.
        
        Args:
            X_train, X_val, X_test (list): Input data for training, validation, and testing.
            D_train, D_val, D_test (Tensor): Duration data.
            E_train, E_val, E_test (Tensor): Event indicator data.
            n_step (int): Maximum number of training steps.
            stop_step (int): Number of steps without improvement before early stopping.
            min_step (int): Minimum number of steps before allowing early stopping.
        """
        self.train()
        
        train_mask_vn = torch.cat([X_train[v].std(1)[None, :] > 0 for v in range(len(X_train))], 0).float()
        
        f = open(self.filename + "_log.txt", "a", buffering=1)
        self.last_update_step = 0
        
        for step in range(n_step):
            self.optim.zero_grad()
            
            h_pred_n, h_pred_vn, mu_nd, logvar_nd = self.forward(X_train, missing_mask_vn=train_mask_vn)
            
            if step >= min_step:
                loss_pred = self.cox_ph_loss(h_pred_n, D_train, E_train)
                alpha = self.alpha
            else:
                self.loss_opt = 1e30
                loss_pred = torch.tensor(0.).cuda()
                alpha = 1e5
            
            KLD = self.KL_Divergence(mu_nd, logvar_nd).mean()
            
            loss_W = sum(self.gamma1 * torch.abs(self.encoder_mu[v].weight).mean() + 
                         self.gamma2 * (self.encoder_mu[v].weight ** 2).mean() 
                         for v in range(self.n_views))
            
            loss_pred_v = sum(self.cox_ph_loss(
                h_pred_vn[v, train_mask_vn[v] == 1.], 
                D_train[train_mask_vn[v] == 1.], 
                E_train[train_mask_vn[v] == 1.])
                for v in range(self.n_views))
            
            loss = loss_pred + loss_W + self.beta * KLD + alpha * loss_pred_v
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1e-4)
            self.optim.step()
            
            # Log training progress
            txt = f"Step: {step}\n"
            txt += f"cox_loss: {loss_pred.item()}\n"
            txt += f"cox_loss_v: {loss_pred_v.item() / h_pred_vn.shape[0]}\n"
            txt += f"KLD: {KLD.item()}\n"
            txt += f"Sparsity loss: {loss_W.item()}\n"
            txt += f"Loss: {loss.item()}\n\n"
            
            if not self.silent:
                print(txt)
            f.write(txt)
            
            print(self.compute_C(X_test, D_test, E_test))
            print(self.compute_C_view(X_test, D_test, E_test))
            
            # Validate and potentially save model
            if step % self.N_validate == self.N_validate - 1:
                loss = self.eval_model(X_val, D_val, E_val)
                
                txt = f"Step: {step}\n"
                txt += f"loss: {loss.item()}\n"
                txt += f"opt loss: {self.loss_opt}\n\n"
                
                print(txt)
                f.write(txt)
                
                if loss < self.loss_opt:
                    self.loss_opt = loss.item()
                    torch.save(self.state_dict(), self.filename)
                    self.last_update_step = step
            
            # Early stopping check
            if step - self.last_update_step >= stop_step and step >= min_step:
                break
    
    def eval_model(self, X, durations, events):
        """
        Evaluate the model on validation data.
        
        Args:
            X (list): Input data.
            durations (Tensor): Time-to-event or censoring time.
            events (Tensor): Event indicators.
        
        Returns:
            Tensor: Computed loss value.
        """
        self.eval()
        h_pred_n, _, _, _ = self.forward(X, non_prop=True)
        loss = self.cox_ph_loss(h_pred_n, durations, events)
        self.train()
        return loss


    def compute_C(self, X, durations, events):
        """
        Compute concordance index for the full model.
        
        Args:
            X (list): Input data.
            durations (Tensor): Time-to-event or censoring time.
            events (Tensor): Event indicators.
        
        Returns:
            float: Concordance index.
        """
        self.eval()
        h_pred_n, _, _, _ = self.forward(X, non_prop=True)
        durations = durations.cpu().data.numpy()
        events = events.cpu().data.numpy()
        h = h_pred_n.cpu().data.numpy()
        self.train()
        return concordance_index(durations, -h, events)
    
    def compute_C_view(self, X, durations, events):
        """
        Compute concordance index for each view.
        
        Args:
            X (list): Input data.
            durations (Tensor): Time-to-event or censoring time.
            events (Tensor): Event indicators.
        
        Returns:
            list: Concordance index for each view.
        """
        self.eval()
        _, h_pred_vn, _, _ = self.forward(X, non_prop=True)
        durations = durations.cpu().data.numpy()
        events = events.cpu().data.numpy()
        h_vn = h_pred_vn.cpu().data.numpy()
        self.train()
        return [concordance_index(durations, -h_vn[v, :], events) for v in range(h_vn.shape[0])]