# VCSM (Variational Consensus Survival Model)

## Overview

This repository contains the code for the Variational Consensus Survival Model (VCSM). VCSM addresses two major limitations of current methods:

1. Traditional multi-view methods excel at identifying shared information across different data views but often neglect the rich complementary information contained in diverse data.
2. Existing approaches struggle to handle missing data views effectively, a common issue in this research area.

VCSM treats the hazard prediction from each data view (i.e., each type of data) as a variational Gaussian distribution. Specifically:

- The mean parameter of this distribution, which models the hazard predicted with each view, is estimated with a linear model.
- The uncertainty in these hazard predictions is modeled through the variance parameter, determined using a multi-layer perceptron (MLP).
- The consensus across various views is given by multiplying the variational Gaussian distributions.

![The model](Figures/Model.png)

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/Chen-Jxiang/VCSM.git
   cd VCSM
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```

3. Install PyTorch with CUDA support:
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

4. Install all dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

## Data Extraction and Preparation

The image features are provided in the `data/` directory. To download and decompress the clinical and omics data, use the following commands:

1. Download the data files:
   ```bash
   wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi
   wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz
   wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct.gz
   wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPM_log2.cct
   ```

2. Decompress the gzipped files:
   ```bash
   gzip -d ./data/Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz
   gzip -d ./data/Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct.gz
   ```

## Training and Evaluating the VCSM Model

### Training

To replicate the experiments in the paper, train the models with 10 seeds using the following command:

```bash
for seed in {0..9}
do
    python ./src/train_VCSM.py --seed $seed
done
```

### Evaluation

The evaluation of the VCSM model is provided in the [evaluate_VCSM.ipynb](./evaluate_VCSM.ipynb) notebook.
