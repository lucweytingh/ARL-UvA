# Reimplementing the Adversarially Reweighted Learning model by Lahoti et al. (2020) to improve fairness without demographics

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

# Datasets
The preprocessed datasets can be found in the folders:
 * ./data/compas/
 * ./data/adult/
 * ./data/lsac/



## Data Preparation
To download the original datasets and have insight the preprosessing process. The following steps can be followed. We have included jupyter notebooks that automate the preprosessing.

### Pre-process COMPAS dataset 
Download the COMPAS dataset from: https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv and save it in the './data/compas/' folder.

Run './data/preprocess_data/CreateCompasDatasetFiles.ipynb' to process the dataset, and create the required files.

### Pre-process UCI Adult dataset 
Download the UCI Adult dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test and save it in the './data/adult/' folder.

Run './data/preprocess_data/CreateUCIAdultDatasetFiles.ipynb' to process the dataset, and create the required files.

### Pre-process  Law School Admissions Council (LSAC) dataset 
Download the  Law School Admissions Council (LSAC) dataset from: http://www.seaphe.org/databases.php and save it in 'data/law_school'.

Run './data/preprocess_data/CreateLawSchoolDatasetFiles.ipynb' to process the dataset, and create the required files.


# Reproducing Experiments
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


# Folder Contents


```bash
.
├── LICENSE
├── README.md
├── argparser.py
├── arl.py
├── baseline.py
├── data
│   ├── README.md
│   ├── data_utils
│   │   ├── dataset_overview_compas.ipynb
│   │   ├── dataset_overview_lsac.ipynb
│   │   └── dataset_overview_uci_adult.ipynb
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── compas
│   │   │   ├── IPS_example_weights_with_label.json
│   │   │   ├── IPS_example_weights_without_label.json
│   │   │   ├── compas-scores-two-years.csv
│   │   │   ├── dataset_stats.json
│   │   │   ├── mean_std.json
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── vocabulary.json
│   │   ├── law_school
│   │   │   ├── IPS_example_weights_with_label.json
│   │   │   ├── IPS_example_weights_without_label.json
│   │   │   ├── dataset_stats.json
│   │   │   ├── lsac.csv
│   │   │   ├── lsac.sas7bdat
│   │   │   ├── mean_std.json
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── vocabulary.json
│   │   └── uci_adult
│   │       ├── IPS_example_weights_with_label.json
│   │       ├── IPS_example_weights_without_label.json
│   │       ├── adult.data
│   │       ├── adult.test
│   │       ├── dataset_stats.json
│   │       ├── mean_std.json
│   │       ├── test.csv
│   │       ├── train.csv
│   │       └── vocabulary.json
│   └── preprocess_data
│       ├── CreateCompasDatasetFiles.ipynb
│       ├── CreateLawSchoolDatasetFiles.ipynb
│       ├── CreateUCIAdultDatasetFiles.ipynb
│       ├── CreateUCISyntheticDataset.ipynb
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-37.pyc
│       │   ├── compas_input.cpython-37.pyc
│       │   ├── law_school_input.cpython-37.pyc
│       │   └── uci_adult_input.cpython-37.pyc
│       └── dataset_overview_compas_lsac.ipynb
├── dataloader.py
├── hyperparameter.py
├── hyperparameters
│   ├── compas.txt
│   ├── law_school.txt
│   └── uci_adult.txt
├── metrics.py
├── requirements.txt
├── results
│   ├── compas_ARL.json
│   ├── compas_baseline.json
│   ├── law_school_ARL.json
│   ├── law_school_baseline.json
│   ├── uci_adult_ARL.json
│   └── uci_adult_baseline.json
├── results.ipynb
├── significance.py
└── train.py

```
