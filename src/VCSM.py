import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F
from pycox.models import utils
from torchtuples import TupleTree

import os
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn import ParameterList
from torch import optim

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from monai.networks.nets import FullyConnectedNet, VarFullyConnectedNet
from torch.nn import BCEWithLogitsLoss, Linear, CrossEntropyLoss, Softmax, Dropout, Sequential, ReLU

from lifelines.utils import concordance_index


class cox_model(torch.nn.Module):    
    def __init__(self, D_dat = [250, ], encoder_var_layer = [ 1024 ] , decoder_layer = [  32 ], 
                 seed = 0, silent = False, lr = 1e-5, 
                 alpha = 0., beta = 1., gamma1 = 0.,  gamma2 = 0., 
                 dropout_encoder = 0., dropout_predictor = 0., p_dropout0 = 0.,
                 N_validate = 50, 
                 filename = "models/model"):

        super(cox_model, self).__init__()        
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.D_dat = D_dat
        
        self.n_views = len(D_dat)
        
        self.alpha = alpha
        self.beta = beta
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        
        self.filename = filename
        self.N_validate = N_validate
        
        if type(dropout_encoder) is float:
            dropout_encoder = [dropout_encoder] * self.n_views

            
        self.dropout = torch.nn.Dropout(p = p_dropout0)
            
        
        self.silent = silent
        
        self.encoder_mu = torch.nn.ModuleList( 
            [Linear(D_dat[iii], 1) 
             for iii in range(self.n_views)
            ] )
        
        self.encoder_var = torch.nn.ModuleList( 
            [FullyConnectedNet(D_dat[iii], 1, encoder_var_layer, act = 'PRELU', dropout = dropout_encoder[iii] ) 
             for iii in range(self.n_views)
            ] )
        

        self.optim = optim.Adam( self.parameters(), 
            lr=lr, betas=(0.0,0.999), eps=1e-8)

        self.loss_opt = 1e30
        
    def reparameterize(self, mu, log_var):
        """
        Re-parameterization tricks for a Guassian distribution. The mean and the logarithm of variance are provided.
        """
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std    
    
    
    def KL_Divergence(self, mu, log_var):
        """
        Computing the KL divergence of a Gaussian distribution to the standard Gaussian distribution with zero mean and unit variance.
        """
        return .5 * ( -1 - log_var + log_var.exp() + mu.pow(2)  )    

    def cox_ph_loss(self, log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
        """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    
        We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
        where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    
        We just compute a cumulative sum, and not the true Risk sets. This is a
        limitation, but simple and fast.
        """
        assert(events.sum() > 0)
        
        idx = durations.sort(descending=True)[1]
        events = events[idx]
        log_h = log_h[idx]
        events = events.view(-1)
        log_h = log_h.view(-1)
        log_h_max = log_h.max()
        # log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)

        log_h_cumsum = torch.log( torch.exp( log_h - log_h_max ).cumsum(0) + eps ) + log_h_max
    
        # return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())
        
        return - ( ( log_h - log_h_cumsum ) * events ).sum() / events.sum()

    
    def PoG(self, mu_vnd, log_var_vnd): 
        """
        Given the means and logarithm of variances of Guasian distributions, the function returns the mean and variances of the product of these Gaussian distributions. 
        """

        log_var_max = log_var_vnd.max()
        
        var_vnd = torch.exp(log_var_vnd - log_var_max)
        var_nd =  1. /  ( 1. / var_vnd) .sum(0) 
        mu_nd = var_nd * ( 1. / var_vnd * mu_vnd).sum(0)
        log_var_nd = torch.log(var_nd)  + log_var_max
        
        return mu_nd, log_var_nd

    def forward(self, X, non_prop = False, missing_mask_vn = None):
        
        N = X[0].shape[0]
        
        mu_vn = torch.zeros( [0, N]  ).cuda()
        logvar_vn = torch.zeros( [0, N]  ).cuda()
        h_pred_vn = torch.zeros( [0, N]  ).cuda()

        
        for vvv in range(self.n_views):
            
            X_vvv = self.dropout( X[vvv] )
            
            mu_vvv = self.encoder_mu[vvv]( X_vvv )
            logvar_vvv = self.encoder_var[vvv]( X_vvv ) 
            
            
            if not missing_mask_vn is None:
                mu_vvv = torch.einsum("nd, n -> nd", mu_vvv, missing_mask_vn[vvv])
                logvar_vvv = torch.einsum("nd, n -> nd", logvar_vvv, missing_mask_vn[vvv]) \
                           + torch.einsum("nd, n -> nd", 5. * torch.ones(logvar_vvv.shape).cuda(), 1 - missing_mask_vn[vvv])
                
                
            mu_vn = torch.concatenate( 
                [mu_vn, 
                  mu_vvv[None, :, 0] ]  )

            logvar_vn = torch.concatenate( 
                [logvar_vn, 
                 logvar_vvv[None, :, 0] ]  )
            
            if non_prop:
                z_vvv = mu_vvv
            else:
                z_vvv = self.reparameterize(mu_vvv, logvar_vvv) 
            
            
            h_pred_vvv = z_vvv
            
            h_pred_vn = torch.concatenate( 
                [h_pred_vn, 
                 h_pred_vvv[None, :, 0] ]  )

            
        mu_n, logvar_n = self.PoG(mu_vn, logvar_vn)
        
        if non_prop:
            z_n = mu_n
        else:
            z_n = self.reparameterize(mu_n, logvar_n)
        
        h_pred_n = z_n.flatten()
        
        return h_pred_n, h_pred_vn, mu_vn, logvar_vn
    
    
    def train_model(self, X_train, D_train, E_train, 
                    X_val, D_val, E_val, 
                    X_test = None, D_test = None, E_test = None, 
                    n_step = 20000, stop_step = 500, min_step = 200):
        
        self.train()
        
        train_mask_vn = torch.cat ( [ X_train[vvv].std(1)[None, :] > 0 for vvv in range(len(X_train)) ] , 0).float()        
        
        f = open( self.filename + "_log.txt", "a", buffering=1 )

        self.last_update_step = 0
        
        for sss in range(n_step):
            
            self.optim.zero_grad()

            h_pred_n, h_pred_vn, mu_nd, logvar_nd = self.forward( X_train, missing_mask_vn = train_mask_vn )
        
            if sss >= min_step:
                loss_pred = self.cox_ph_loss(h_pred_n, D_train, E_train)
                alpha = self.alpha
            else:
                self.loss_opt = 1e30

                loss_pred = torch.tensor(0.).cuda()
                alpha = 1e5
            

            KLD = self.KL_Divergence(mu_nd, logvar_nd).mean()                    
            
            
            loss_W = torch.tensor(0.).cuda()
            for vvv in range( self.n_views ):
                loss_W += self.gamma1 * torch.abs(self.encoder_mu[vvv].weight).mean() \
                          + self.gamma2 * (self.encoder_mu[vvv].weight ** 2).mean()
                
            loss_pred_v = 0
            for vvv in range( self.n_views ):
                loss_pred_v += self.cox_ph_loss(
                        h_pred_vn[vvv, train_mask_vn[vvv] == 1.], 
                        D_train[train_mask_vn[vvv] == 1.], 
                        E_train[train_mask_vn[vvv] == 1.])
                    

            
            
            loss = loss_pred + loss_W + self.beta * KLD  + alpha * loss_pred_v

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1e-4)
            self.optim.step()
    
            txt = "Step: {}\n".format(sss)
            txt += "cox_loss: {}\n".format( loss_pred.item() )
            txt += "cox_loss_v: {}\n".format( loss_pred_v.item() / h_pred_vn.shape[0] )
            txt += "KLD: {}\n".format( KLD.item() )
            txt += "Sparsity loss: {}\n".format( loss_W.item() )
            txt += "Loss: {}\n\n".format(loss.item())        
            
            if not self.silent:
                print(txt)            
            f.write(txt)    

            print( self.compute_C (X_test, D_test, E_test) )
            print( self.compute_C_view(X_test, D_test, E_test) )
    
    
            if sss % self.N_validate == self.N_validate - 1:
                # Save some results in the log file and print them.



                # Evaludate the model with the validation dataset and return the r-squared.
                loss = self.eval_model( X_val, D_val, E_val)

                txt = "Step: {}\n".format(sss)
                txt += "loss:{}\n".format(loss.item())
                txt += "opt loss:{}\n\n".format(self.loss_opt)

                print(txt)

                # Saving the model only when the evaluated results are better than the current optimum.
                if loss < self.loss_opt:
                    self.loss_opt = loss.item()
                    torch.save( self.state_dict(), self.filename )
                    self.last_update_step = sss
                
                f.write(txt)    

            if sss - self.last_update_step >= stop_step and sss >= min_step:
                break    
    
    def eval_model(self, X, durations, events):
        self.eval()
        h_pred_n, h_pred_vn, mu_vnd, logvar_vnd = self.forward(X, non_prop = True)
        loss = self.cox_ph_loss(h_pred_n, durations, events)
        self.train()

        return loss

    def compute_C(self, X, durations, events):
        self.eval()
        h_pred_n, h_pred_vn, mu_vnd, logvar_vnd = self.forward(X, non_prop = True)

        durations = durations.cpu().data.numpy()
        events = events.cpu().data.numpy()
        h = h_pred_n.cpu().data.numpy()
        self.train()

        return concordance_index(durations, -h, events)                
    
    def compute_C_view(self, X, durations, events):
        self.eval()
        h_pred_n, h_pred_vn, mu_vnd, logvar_vnd = self.forward(X, non_prop = True)

        durations = durations.cpu().data.numpy()
        events = events.cpu().data.numpy()
        h_vn = h_pred_vn.cpu().data.numpy()
        self.train()

        return [concordance_index(durations, -h_vn[vvv, :], events) for vvv in range(h_vn.shape[0])]