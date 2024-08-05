# VCSM
This repository contains the code for Variational Consensus SurvivalModel (VCSM). VCSM overcomes two major limitations of current
methods: 1) while traditional multi-view methods excel at identifying shared information across different data views, they often neglect the rich complementary information contained in the diverse data; and 2) existing approaches struggle to handle missing data views effectively,
a common issue in this research area. VCSM treats the hazard prediction from each data view (i.e., each type of data) as a variational Gaussian distribution. Specifically, the mean parameter of this distribution that models the hazard predicted with each view is estimated with a linear model, while the uncertainty in these hazard predictions is modelled through the variance parameter, which is determined using a multi-layer perceptron (MLP). The consensus across various views is given by multiplying the variational Gaussian distributions.

![The model](Figures/Model.png)

## Installation

To use this project, follow these steps:

### 1. Clone the repository:

git clone https://github.com/Chen-Jxiang/VCSM.git<br/>
cd VCSM

### 2. Create a virtual environment (optional but recommended):
python -m venv venv<br/>
source venv/bin/activate  # On Windows, use venv\Scripts\activate

### 3. Install PyTorch with CUDA support:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

### 4. Install all dependencies by running:
pip install -r requirements.txt

## Data Extraction and Preparation

The image features extracted are provided in the `data/` directory. To download and decompress the clinical and omics data, use the following commands:
### 1. Download the data files:
wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi<br/>
wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz<br/>
wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct.gz<br/>
wget -P ./data/ https://linkedomics.org/data_download/TCGA-SKCM/Human__TCGA_SKCM__BDGSC__miRNASeq__HS_miR__01_28_2016__BI__Gene__Firehose_RPM_log2.cct
### 2.Decompress the gzipped files:
gzip -d ./data/Human__TCGA_SKCM__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct.gz<br/>
gzip -d ./data/Human__TCGA_SKCM__JHU_USC__Methylation__Meth450__01_28_2016__BI__Gene__Firehose_Methylation_Prepocessor.cct.gz<br/>
