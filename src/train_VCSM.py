import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from importlib import reload
import sys
import argparse

# Add the '../src' directory to the Python path
sys.path.append("../src")

# Import custom modules
import VCSM
import SKCM_dataset


def train_model(lr, 
                encoder_layer, predictor_layer, 
                alpha, beta, gamma1, gamma2, 
                top_K_img, top_K_RNA, top_K_meth, top_K_miRNA, 
                seed, n_step, stop_step, min_step):

    """
    Train the VCSM model with the given parameters.
    
    Args:
        lr (float): Learning rate
        encoder_layer (list): Encoder layer dimensions
        predictor_layer (list): Predictor layer dimensions
        alpha, beta, gamma1, gamma2 (float): Model hyperparameters
        top_K_img, top_K_RNA, top_K_meth, top_K_miRNA (int): Feature selection parameters
        seed (int): Random seed
        n_step, stop_step, min_step (int): Training step parameters
    """    
    
    # Create models directory if it doesn't exist
    os.makedirs("./models/", exist_ok=True)
    
    # Generate filename for the model
    filename = "./models/VCSM_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        encoder_layer, predictor_layer, alpha, beta, gamma1, gamma2, 
        lr, top_K_img, top_K_RNA, top_K_meth, top_K_miRNA, seed)

    # Load and preprocess data
    X_train, D_train, E_train, X_test, D_test, E_test = SKCM_dataset.load_data(
    seed = seed, 
    top_K_img = top_K_img, top_K_RNA = top_K_RNA, top_K_meth = top_K_meth, top_K_miRNA = top_K_miRNA)

    X_train = [XXX.cuda() for XXX in X_train]
    D_train = D_train.cuda()
    E_train = E_train.cuda()
    X_test = [XXX.cuda() for XXX in X_test]
    D_test = D_test.cuda()
    E_test = E_test.cuda()

    
    # Initialize the model
    model = VCSM.CoxModel(
        D_dat=[x.shape[1] for x in X_train], 
        encoder_var_layer=encoder_layer, decoder_layer=predictor_layer,
        N_validate=1, alpha=alpha, beta=beta, gamma1=gamma1, gamma2=gamma2,
        filename=filename,
        silent=True, 
        lr=lr
    ).cuda()
    
    
    model.train_model(X_train, D_train, E_train, 
                      X_train, D_train, E_train, 
                      X_test=X_test, D_test=D_test, E_test=E_test, 
                      n_step=n_step, stop_step=stop_step, min_step=min_step)

    print("Save Model", filename)

   


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train VCSM model for multi-omics data integration.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--encoder_layer', default="128,64,32", type=str, help='Encoder layer dimensions (comma-separated)')
    parser.add_argument('--predictor_layer', default="", type=str, help='Predictor layer dimensions (comma-separated)')
    parser.add_argument('--alpha', default=0., type=float, help='Alpha hyperparameter')
    parser.add_argument('--beta', default=10., type=float, help='Beta hyperparameter')
    parser.add_argument('--gamma1', default=0., type=float, help='Gamma1 hyperparameter')
    parser.add_argument('--gamma2', default=0., type=float, help='Gamma2 hyperparameter')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--top_K_img', default=250, type=int, help='Top K features for image data')
    parser.add_argument('--top_K_RNA', default=500, type=int, help='Top K features for RNA data')
    parser.add_argument('--top_K_meth', default=500, type=int, help='Top K features for methylation data')
    parser.add_argument('--top_K_miRNA', default=50, type=int, help='Top K features for miRNA data')
    parser.add_argument('--n_step', default=10000, type=int, help='Number of training steps')
    parser.add_argument('--stop_step', default=5, type=int, help='Early stopping steps')
    parser.add_argument('--min_step', default=500, type=int, help='Minimum number of training steps')
    parser.add_argument('--drop_bin', action=argparse.BooleanOptionalAction, default=True, help='Whether to drop binary features')
    
    args = parser.parse_args()

    # Convert comma-separated strings to lists of integers
    encoder_layer = [] if args.encoder_layer == "" else [int(x) for x in args.encoder_layer.split(",")]
    predictor_layer = [] if args.predictor_layer == "" else [int(x) for x in args.predictor_layer.split(",")]


    train_model(args.lr, encoder_layer, predictor_layer,
                args.alpha, args.beta, args.gamma1, args.gamma2, 
                args.top_K_img, args.top_K_RNA, args.top_K_meth, args.top_K_miRNA, 
                args.seed, args.n_step, args.stop_step, args.min_step)
