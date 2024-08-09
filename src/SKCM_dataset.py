import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.preprocessing import OneHotEncoder

def load_data(seed=0, top_K_img=None, top_K_RNA=None, top_K_meth=None, top_K_miRNA=None):
    """
    Load and preprocess SKCM (Skin Cutaneous Melanoma) data from various sources.
    
    :param seed: Random seed for reproducibility
    :param top_K_img: Number of top image features to select
    :param top_K_RNA: Number of top RNA-seq features to select
    :param top_K_meth: Number of top methylation features to select
    :param top_K_miRNA: Number of top miRNA features to select
    :return: Tuple of training and testing data (features, durations, and events)
    """
    np.random.seed(seed)
    
    # Load clinical data
    df_clinical = pd.read_csv("./data/Human__TCGA_SKCM__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi", 
                              delimiter="\t").set_index("attrib_name").T
    
    # Load and process cell statistics data
    df_cell = pd.read_csv("./data/cell_stats.csv")    
    df_cell = df_cell.set_index("sid").astype(float)    
    
    # Select top K image features if specified
    if top_K_img is not None:
        p_vals = np.load(f"./data/cox_p-values/WSI/{seed}.npy")
        img_idx = np.argsort(p_vals)[:top_K_img]
        df_cell = pd.merge(left=df_cell.iloc[:, img_idx], right=df_cell.iloc[:, -4:], left_index=True, right_index=True)
    
    # Load and process RNA-seq data
    RNAseq = pd.read_csv("./data/Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct", 
                         sep="\t").set_index("attrib_name").T
    RNAseq_rename = {ccc: f"RNA_{ccc}" for ccc in RNAseq.columns}
    RNAseq = RNAseq.rename(RNAseq_rename, axis=1)

    # Select top K RNA-seq features if specified
    if top_K_RNA is not None:
        p_vals = np.load(f"./data/cox_p-values/RNAseq/{seed}.npy")
        RNA_idx = np.argsort(p_vals)[:top_K_RNA]
        RNAseq = RNAseq.iloc[:, RNA_idx]

    # Load and process methylation data
    meth = pd.read_csv("./data/Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct", 
                       sep="\t").set_index("attrib_name").T     
    meth_rename = {ccc: f"meth_{ccc}" for ccc in meth.columns}
    meth = meth.rename(meth_rename, axis=1)

    # Select top K methylation features if specified
    if top_K_meth is not None:
        p_vals = np.load(f"./data/cox_p-values/meth/{seed}.npy")
        meth_idx = np.argsort(p_vals)[:top_K_meth]
        meth = meth.iloc[:, meth_idx]

    # Load and process miRNA data
    miRNASeq = pd.read_csv("./data/Human__TCGA_SKCM__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPM_log2.cct", 
                           sep="\t").set_index("attrib_name").T     
    miRNASeq_rename = {ccc: f"miRNA_{ccc}" for ccc in miRNASeq.columns}
    miRNASeq = miRNASeq.rename(miRNASeq_rename, axis=1)
    
    # Select top K miRNA features if specified
    if top_K_miRNA is not None:
        p_vals = np.load(f"./data/cox_p-values/miRNA/{seed}.npy")
        miRNA_idx = np.argsort(p_vals)[:top_K_miRNA]
        miRNASeq = miRNASeq.iloc[:, miRNA_idx]    

    # Merge all data sources
    df_merged = pd.merge(left=df_cell, right=RNAseq, left_index=True, right_index=True, how="outer")
    df_merged = pd.merge(left=df_merged, right=meth, left_index=True, right_index=True, how="outer")    
    df_merged = pd.merge(left=df_merged, right=miRNASeq, left_index=True, right_index=True, how="outer")    
    df_merged = df_merged.fillna(0)

    # Add duration and events from clinical data
    df_merged["duration"] = df_clinical.loc[df_merged.index, "overall_survival"].astype(float)    
    df_merged["events"] = df_clinical.loc[df_merged.index, "status"].astype(float)    
    
    # Get common sample IDs across all data sources
    sid_set = set.intersection(set(df_cell.index), set(RNAseq.index), set(meth.index), set(miRNASeq.index))

    # Split samples into positive and negative events
    pos_sid = [ppp for ppp in sid_set if df_merged.loc[ppp, "events"]]
    neg_sid = [ppp for ppp in sid_set if not df_merged.loc[ppp, "events"]]
    
    # Calculate train-test split sizes
    N_train_pos = int(0.8 * len(pos_sid))
    N_train_neg = int(0.8 * len(neg_sid))

    np.random.shuffle(pos_sid)
    np.random.shuffle(neg_sid)    

    # Save or load sample IDs for reproducibility
    os.makedirs("./data/sids", exist_ok=True)
    
    if not os.path.isfile(f"./data/sids/pos_sid{seed}.txt"):
        with open(f"./data/sids/pos_sid{seed}.txt", "w") as f:
            f.write("\n".join(pos_sid))
        with open(f"./data/sids/neg_sid{seed}.txt", "w") as f:
            f.write("\n".join(neg_sid))
    else:
        pos_sid = list(np.loadtxt(f"./data/sids/pos_sid{seed}.txt", dtype=str))
        neg_sid = list(np.loadtxt(f"./data/sids/neg_sid{seed}.txt", dtype=str))
    
    # Split data into training and testing sets
    train_ids = pos_sid[:N_train_pos] + neg_sid[:N_train_neg] + list(set(df_merged.index[~df_merged["duration"].isnull()]) - sid_set)
    test_ids = pos_sid[N_train_pos:] + neg_sid[N_train_neg:]
    df_train = df_merged.loc[train_ids]
    df_test = df_merged.loc[test_ids]

    # Prepare training data
    X_cell_train = torch.tensor(df_train[df_cell.columns].to_numpy()).float()
    X_RNA_train = torch.tensor(df_train[RNAseq.columns].to_numpy()).float()
    X_meth_train = torch.tensor(df_train[meth.columns].to_numpy()).float()
    X_miRNA_train = torch.tensor(df_train[miRNASeq.columns].to_numpy()).float()
    D_train = torch.tensor(df_train["duration"].to_numpy()).float()
    E_train = torch.tensor(df_train["events"].to_numpy()).float()

    # Prepare testing data
    X_cell_test = torch.tensor(df_test[df_cell.columns].to_numpy()).float()
    X_RNA_test = torch.tensor(df_test[RNAseq.columns].to_numpy()).float()
    X_meth_test = torch.tensor(df_test[meth.columns].to_numpy()).float()
    X_miRNA_test = torch.tensor(df_test[miRNASeq.columns].to_numpy()).float()
    D_test = torch.tensor(df_test["duration"].to_numpy()).float()
    E_test = torch.tensor(df_test["events"].to_numpy()).float()

    return [X_cell_train, X_RNA_train, X_meth_train, X_miRNA_train], D_train, E_train, \
           [X_cell_test, X_RNA_test, X_meth_test, X_miRNA_test], D_test, E_test