import numpy as np
import pandas as pd
import os
import sys
import warnings
from lifelines import CoxPHFitter


# Add the src directory to the Python path
sys.path.append("./src/")

import SKCM_dataset

def compute_p_values(view, seed):
    """
    Compute p-values for Cox proportional hazards model for a given view and seed.
    
    :param view: Index of the data view (0: WSI, 1: RNAseq, 2: meth, 3: miRNA)
    :param seed: Random seed for data splitting
    :return: List of p-values
    """
    p_values = []
    
    # Load data for the current seed
    X_train, D_train, E_train, X_test, D_test, E_test = SKCM_dataset.load_data(seed=seed)
    
    # Extract data for the specified view
    XXX = X_train[view].cpu().data.numpy()
    D_train = D_train.cpu().data.numpy()
    E_train = E_train.cpu().data.numpy()
    
    # Filter out features with zero standard deviation
    idx_used = XXX.std(1) > 0
    XXX = XXX[idx_used]
    D_train = D_train[idx_used]
    E_train = E_train[idx_used]
    
    for ddd in range(XXX.shape[1]):
        # Prepare data for Cox model
        data_dict = {"X": XXX[:, ddd], "D": D_train, "E": E_train}
        df = pd.DataFrame(data_dict)
        
        try:
            # Fit Cox proportional hazards model
            cph = CoxPHFitter()
            cph.fit(df, duration_col='D', event_col='E')
            p_values.append(cph.summary.loc["X", "p"])
        except:
            # If fitting fails, assign a p-value of 1
            p_values.append(1.)
        
        if ddd % 100 == 0:
            print(f"Feature: {ddd}/{XXX.shape[1]}")
    
    return p_values

        
        
        
if __name__ == "__main__":
    # Create directories for storing results
    result_dirs = [
        "./data/cox_p-values",
        "./data/cox_p-values/RNAseq",
        "./data/cox_p-values/meth",
        "./data/cox_p-values/miRNA",
        "./data/cox_p-values/WSI"
    ]
    
    for directory in result_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Compute and save p-values for each view and seed
    with warnings.catch_warnings(action="ignore"):
        for seed in range(10):
            for view, view_name in enumerate(["WSI", "RNAseq", "meth", "miRNA"]):
                p_values = compute_p_values(view, seed)
                np.save(f"./data/cox_p-values/{view_name}/{seed}.npy", p_values)
                print(f"Saved p-values for {view_name}, seed {seed}")

