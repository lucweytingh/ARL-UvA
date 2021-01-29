# Fairness without Demographics through Adversarially Reweighted Learning #



## Contributors ##

 * Casper Wortmann
    * casper.wortmann@student.uva.nl
 * Jardenna Mohazzab
    * jardenna.nl@gmail.com
 * Luc Weytingh
    * luc.weytingh@student.uva.nl
 * Barbera Brocades Zaalberg
    * barbara.bz@outlook.com


<br />

## Data Preparation ##

The data provided in the ```'./FACT/dataset_folder/data/toy_data``` directory is dummy, and is only for testing the code. 
For meaningful results, please follow the steps below.


### Pre-process COMPAS dataset and create train and test files: ###
Download the COMPAS dataset from: 

https://github.com/propublica/compas-analysis/blob/master/compas-scores-two-years.csv 

and save it in the ```'./FACT/dataset_folder/dataset_folder/data/compas'``` folder.

Run ```'./FACT/data_utils/CreateCompasDatasetFiles.ipynb'``` notebook to process the dataset, and create files required for training.



### Pre-process UCI Adult (Census Income) dataset and create train and test files: ###

Download the Adult train and test data files can be downloaded from: 

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data 

https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test 


and save them in the ```'./FACT/dataset_folder/dataset_folder/data/uci_adult'``` folder.

Run ```'./FACT/data_utils/CreateLawSchoolDatasetFiles.ipynb'``` notebook to process the dataset, and create files required for training.





### Pre-process Law School Admissions Council (LSAC) Dataset and create train and test files: ###
Download the Law School dataset from: (http://www.seaphe.org/databases.php), convert SAS file to CSV, 

and save it in the ```./FACT/dataset_folder/data/law_school``` folder.

Run CreateLawSchoolDatasetFiles.ipynb to process the dataset, and create files required for training.




### Generate synthetic datasets used in the paper: ###
To generate various synthetic datasets used in the paper run ```'./FACT/data_utils/CreateUCISyntheticDataset.ipynb'``` notebook.

<br />


## Folder contents ##


```bash
.
├── LICENSE
├── README.md
├── dataset_folder
│   ├── README.md
│   ├── data
│   │   ├── __init__.py
│   │   ├── compas
│   │   │   ├── IPS_example_weights_with_label.json
│   │   │   ├── IPS_example_weights_without_label.json
│   │   │   ├── compas-scores-two-years.csv
│   │   │   ├── mean_std.json
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── vocabulary.json
│   │   ├── law_school
│   │   │   ├── lsac.csv
│   │   │   ├── lsac.sas7bdat
│   │   │   ├── mean_std.json
│   │   │   ├── test.csv
│   │   │   ├── train.csv
│   │   │   └── vocabulary.json
│   │   ├── toy_data
│   │   └── uci_adult
│   │       ├── IPS_example_weights_with_label.json
│   │       ├── IPS_example_weights_without_label.json
│   │       ├── adult.data
│   │       ├── adult.test
│   │       ├── mean_std.json
│   │       └── vocabulary.json
│   ├── data_utils
│   │   ├── CreateCompasDatasetFiles.ipynb
│   │   ├── CreateLawSchoolDatasetFiles.ipynb
│   │   ├── CreateUCIAdultDatasetFiles.ipynb
│   │   ├── CreateUCISyntheticDataset.ipynb
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── compas_input.cpython-37.pyc
│   │   │   ├── law_school_input.cpython-37.pyc
│   │   │   └── uci_adult_input.cpython-37.pyc
│   │   ├── adult.data
│   │   ├── adult.test
│   │   ├── compas_input.py
│   │   ├── law_school_input.py
│   │   └── uci_adult_input.py
│   ├── dataset_readers
│   ├── evaluators
│   ├── models
│   └── predictors
├── metrics.py
├── hyperparameter.py
├── hyperparameters
│   ├── compas.txt
│   ├── law_school.txt
│   └── uci_adult.txt
├── pytorch.job
├── requirements.txt
├── arl.py
├── significance
├── train.py
└── utils.py
```


