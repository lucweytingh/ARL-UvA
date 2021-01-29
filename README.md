# Fairness without Demographics through Adversarially Reweighted Learning (analysis of reproducibility)

## Abstract
Literature on Machine Learning (ML) Fairness often assumes that protected features such as race and sex are present in the dataset. However, in practice these features are often not included because of privacy rules and regulations. Therefore, Lahoti et al. \cite{lahoti2020fairness} studied the following question: “How can we train a ML model to improve fairness when we do not know the protected group memberships?”. Reproducibility is a key characteristic of good science, therefore this report is reproduced. This research has three objectives: (1) to replicate the results by Lahoti et al, (2) to test its reproducibility, for which the ARL algorithm will be reimplemented in PyTorch and (3) to test the significance and noise sensitivity of the model.

## Authors
* L.R. Weytingh
* J. Mohazzab
* C.A. Wortmann
* B. Brocades Zaalberg

## Setup
### Prerequisitess
```
Install python and miniconda
```

### Requirements
Initiate conda environment
```
conda create -n arl python=3.6
source activate arl
```

Install the necissary dependencies
```
pip install -r requirements.txt
```

# Reproducing Experiments
## Data Preparation

### Pre-process COMPAS dataset 
Download the COMPAS dataset from: https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv and save it in 'data/compas/'.

Run '???.ipynb' notebook to process dataset, and create files.

### Pre-process UCI Adult dataset 
Download the UCI Adult dataset from: https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test and save it in 'data/adult/'.

Run '???.ipynb' notebook to process dataset, and create files.

### Pre-process  Law School Admissions Council (LSAC) dataset 
Download the  Law School Admissions Council (LSAC) dataset from: http://www.seaphe.org/databases.php and save it in 'data/compas'.

Run '???.ipynb' notebook to process dataset, and create files.


## Reproduce experiments
Reproduce the experiments by running:
```
python train.py
```

Reproduce the hyperparameter search by running:
```
python hyperparameter.py
```


## Citation
```
@article{lahoti2020fairness,
  title={Fairness without demographics through adversarially reweighted learning},
  author={Lahoti, Preethi and Beutel, Alex and Chen, Jilin and Lee, Kang and Prost, Flavien and Thain, Nithum and Wang, Xuezhi and Chi, Ed H},
  journal={arXiv preprint arXiv:2006.13114},
  year={2020}
}
```

