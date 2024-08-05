import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
import os

from sklearn.preprocessing import OneHotEncoder

data_dir = '../data/'

def load_data(seed = 0, drop_bin = True, drop_kurtosis = False, drop_entropy = False, drop_skew = False,
             top_K_img = None, top_K_RNA = None, top_K_meth = None, top_K_miRNA = None, return_sex_age = False, concat_sex_age = False):
    
    np.random.seed(seed)
    
    
    df_clinical = pd.read_csv(data_dir + "Human__TCGA_SKCM__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi", 
                          delimiter = "\t").set_index("attrib_name").T
   
    
    df_cell = pd.read_csv("/N/slate/jch26/rep/data_linkedomics/SKCM/cell_stats.csv")    
    df_cell["sid"] = [ iii.replace("-", ".") for iii in df_cell["sid"] ]
    df_cell = df_cell.set_index("sid").astype(float)    
    
    for iii in df_cell.columns: 
        if "bin" in iii and drop_bin:
            df_cell = df_cell.drop(iii, axis=1)
        elif "kurtosis" in iii and drop_kurtosis:
            df_cell = df_cell.drop(iii, axis=1)
        elif "entropy" in iii and drop_entropy:
            df_cell = df_cell.drop(iii, axis=1)
        elif "skew" in iii and drop_skew:
            df_cell = df_cell.drop(iii, axis=1)    

    if not top_K_img is None:
        p_vals = np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM_.8/corrected_p-values/image/{}.txt".format( seed ) )
        img_idx = np.argsort(p_vals)[ : top_K_img ]
        df_cell = pd.merge(left = df_cell.iloc[:, img_idx], right = df_cell.iloc[:, -4:], left_index = True, right_index = True)
        
            
    RNAseq = pd.read_csv(
        data_dir + "Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct", 
                         sep= "\t").set_index("attrib_name").T

    RNAseq_rename = {ccc: "RNA_" + ccc for ccc in RNAseq.columns}
    RNAseq =  RNAseq.rename(RNAseq_rename, axis=1)

    
    if not top_K_RNA is None:
        p_vals = np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM_.8/cox_p-values/RNAseq/{}.txt".format( seed ) )
        RNA_idx = np.argsort(p_vals)[ : top_K_RNA ]
        RNAseq = RNAseq.iloc[:, RNA_idx]
        

    meth = pd.read_csv(
    data_dir + "Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct", 
                         sep= "\t").set_index("attrib_name").T     

    meth_rename = {ccc: "meth_" + ccc for ccc in meth.columns}
    meth =  meth.rename(meth_rename, axis=1)

    if not top_K_meth is None:
        p_vals = np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM_.8/cox_p-values/meth/{}.txt".format( seed ) )
        meth_idx = np.argsort(p_vals)[ : top_K_meth ]
        meth = meth.iloc[:, meth_idx]

    

    
    miRNASeq = pd.read_csv(
    data_dir + "Human__TCGA_SKCM__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPM_log2.cct", 
                         sep= "\t").set_index("attrib_name").T     

    miRNASeq_rename = {ccc: "miRNA_" + ccc for ccc in miRNASeq.columns}
    miRNASeq =  miRNASeq.rename(miRNASeq_rename, axis=1)
    
    if not top_K_miRNA is None:
        p_vals = np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM_.8/cox_p-values/miRNA/{}.txt".format( seed ) )
        miRNA_idx = np.argsort(p_vals)[ :top_K_miRNA ]
        miRNASeq = miRNASeq.iloc[:, miRNA_idx]    
    

    
    
    df_merged = pd.merge(left = df_cell, right = RNAseq, left_index = True, right_index = True, how = "outer")
    df_merged = pd.merge(left = df_merged, right = meth, left_index = True, right_index = True, how = "outer")    
    df_merged = pd.merge(left = df_merged, right = miRNASeq, left_index = True, right_index = True, how = "outer")    
    df_merged = df_merged.fillna(0)


    df_merged["age"] = df_clinical.loc[df_merged.index, "years_to_birth"].astype(float)
    df_merged["sex"] = (df_clinical.loc[df_merged.index, "gender"] == "female").astype(float)    
    df_merged["duration"] = df_clinical.loc[df_merged.index, "overall_survival"].astype(float)    
    df_merged["events"] = df_clinical.loc[df_merged.index, "status"].astype(float)    
    
    sid_set = set.intersection(set(df_cell.index), set(RNAseq.index), set(meth.index), set(miRNASeq.index))


    pos_sid = []
    neg_sid = []

    for ppp in sid_set:
        if df_merged.loc[ppp, "events"]:
            pos_sid.append(ppp)
        else:
            neg_sid.append(ppp)
    
    
    N_train_pos = int( .8 * len(pos_sid)) 
    N_train_neg = int( .8 * len(neg_sid)) 

    np.random.shuffle(pos_sid)
    np.random.shuffle(neg_sid)    
    
    if not os.path.isfile("/N/slate/jch26/rep/data_linkedomics/SKCM/sids/pos_sid{}.txt".format(seed)):
        f = open("/N/slate/jch26/rep/data_linkedomics/SKCM/sids/pos_sid{}.txt".format(seed), "w")
        for sss in pos_sid:
            f.write(sss + "\n")
        f.close()
        f = open("/N/slate/jch26/rep/data_linkedomics/SKCM/sids/neg_sid{}.txt".format(seed), "w")
        for sss in neg_sid:
            f.write(sss + "\n")
        f.close()
    else:
        pos_sid = list( np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM/sids/pos_sid{}.txt".format(seed), str) )
        neg_sid = list( np.loadtxt("/N/slate/jch26/rep/data_linkedomics/SKCM/sids/neg_sid{}.txt".format(seed), str) )
    
    
    
    df_train = df_merged.loc[pos_sid[:N_train_pos] + neg_sid[:N_train_neg] + \
                             list( set(df_merged.index[~df_merged["duration"].isnull()]) - sid_set )]
    df_test = df_merged.loc[pos_sid[N_train_pos:] + neg_sid[N_train_neg:]]


    X_cell_train = torch.tensor( df_train[df_cell.columns].drop(["sex", "age", "duration", "events"], axis = 1).to_numpy() ).cuda().float()
    X_RNA_train = torch.tensor( df_train[RNAseq.columns].to_numpy() ).cuda().float()
    X_meth_train = torch.tensor( df_train[meth.columns].to_numpy() ).cuda().float()
    X_miRNA_train = torch.tensor( df_train[miRNASeq.columns].to_numpy() ).cuda().float()
    X_sex_age_train = torch.tensor( df_train[["sex", "age"]].to_numpy() ).cuda().float()
    
    D_train = torch.tensor( df_train["duration"].to_numpy() ).cuda().float()
    E_train = torch.tensor( df_train["events"].to_numpy() ).cuda().float()

    X_cell_test = torch.tensor( df_test[df_cell.columns].drop(["sex", "age", "duration", "events"], axis = 1).to_numpy() ).cuda().float()
    X_RNA_test = torch.tensor( df_test[RNAseq.columns].to_numpy() ).cuda().float()
    X_meth_test = torch.tensor( df_test[meth.columns].to_numpy() ).cuda().float()
    X_miRNA_test = torch.tensor( df_test[miRNASeq.columns].to_numpy() ).cuda().float()
    X_sex_age_test = torch.tensor( df_test[["sex", "age"]].to_numpy() ).cuda().float()

    
    D_test = torch.tensor( df_test["duration"].to_numpy() ).cuda().float()
    E_test = torch.tensor( df_test["events"].to_numpy() ).cuda().float()
    
    if concat_sex_age:
        X_cell_train = torch.cat([X_cell_train, X_sex_age_train], 1)
        X_RNA_train = torch.cat([X_RNA_train, X_sex_age_train], 1)
        X_meth_train = torch.cat([X_meth_train, X_sex_age_train], 1)
        X_miRNA_train = torch.cat([X_miRNA_train, X_sex_age_train], 1)

        X_cell_test = torch.cat([X_cell_test, X_sex_age_test], 1)
        X_RNA_test = torch.cat([X_RNA_test, X_sex_age_test], 1)
        X_meth_test = torch.cat([X_meth_test, X_sex_age_test], 1)
        X_miRNA_test = torch.cat([X_miRNA_test, X_sex_age_test], 1)
        
        
    if return_sex_age:
        return [X_cell_train, X_RNA_train, X_meth_train, X_miRNA_train, X_sex_age_train], D_train, E_train, \
           [X_cell_test, X_RNA_test, X_meth_test, X_miRNA_test, X_sex_age_test], D_test, E_test
    else:
        return [X_cell_train, X_RNA_train, X_meth_train, X_miRNA_train], D_train, E_train, \
           [X_cell_test, X_RNA_test, X_meth_test, X_miRNA_test], D_test, E_test


    