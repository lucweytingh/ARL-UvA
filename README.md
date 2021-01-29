# Fairness without Demographics through Adversarially Reweighted Learning (analysis of reproducibility)

## Abstract
This repository provides a PyTorch implementation of the Adversarially Reweighted Learning (ARL) model, as proposed by Lahoti et al.  

## Authors
* J. Mohazzab
* L.R. Weytingh
* C.A. Wortmann
* B. Brocades Zaalberg

## Setup
### Prerequisitess
```
Python and miniconda
```

### Requirements
Initiate conda environment
```
conda create -n arl_uva python=3.8
conda activate arl_uva
```

Install the necessary dependencies
```
pip install -r requirements.txt
```

# Reproducing Experiments
## Data Preparation

### Pre-process COMPAS dataset 
Download the COMPAS dataset from: https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv and save it in 'data/compas/'.

Run 'data/preprocess_data/CreateCompasDatasetFiles.ipynb' to process the dataset, and create the required files.

### Pre-process UCI Adult dataset 
Download the UCI Adult dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test and save it in 'data/adult/'.

Run 'data/preprocess_data/CreateUCIAdultDatasetFiles.ipynb' to process the dataset, and create the required files.

### Pre-process  Law School Admissions Council (LSAC) dataset 
Download the  Law School Admissions Council (LSAC) dataset from: http://www.seaphe.org/databases.php and save it in 'data/compas'.

Run 'data/preprocess_data/CreateLawSchoolDatasetFiles.ipynb' to process the dataset, and create the required files.


## Reproduce experiments
Reproduce the experiments by running:
```
results.ipynb
```

Reproduce the hyperparameter search by running:
```
python hyperparameter.py
```


## Citation of the original paper by Lahoti et al.
```
@article{lahoti2020fairness,
  title={Fairness without demographics through adversarially reweighted learning},
  author={Lahoti, Preethi and Beutel, Alex and Chen, Jilin and Lee, Kang and Prost, Flavien and Thain, Nithum and Wang, Xuezhi and Chi, Ed H},
  journal={arXiv preprint arXiv:2006.13114},
  year={2020}
}
```

