# Predicting-Alzheimers

This project aims to investigate the capability of machine learning models to predict the progression of long term diseases.

We look specifically at Alzheimer's disease.

## Project Setup

All of the scripts etc are in the top level.

It's **super important** you make a directory called `data`, and place `TADPOLE_D1_D2.csv` in here, once you download the data from the [ADIS website](https://ida.loni.usc.edu/login.jsp).

A.k.a. follow the directory structure below.

```
.
├── ...
├── data
│   ├── TADPOLE_D1_D2.csv
└── ...    
```

## Generating the benchmark datasets

Run `create_dataset.py`. Make sure that `csv_path` is set correctly.

```
csv_path = './data/TADPOLE_D1_D2.csv'
df = pd.read_csv(csv_path)
```


Two datasets are created: `./data/dataset_standard.csv` and `./data/dataset_benchmark.csv` which correspond to the datset with **standard preprocessing** and **benchmark preprocessing** applied respectively.

Standard preprocessing:
- Maps diagnosis
- Adds age at exam
- Converts csf features to numeric
- Imputes missing diagnosis values where the diagnosis before and after is the same
- Adds the target features

Benchmark preprocessing:
- Drops NaNs in `diagnosis` and `target_diagnosis` (needed for next step)
- Encodes `diagnosis` and `target_diagnosis`
- Drops entries where `APOE` is equal to 2

The resulting datasets **contain all features** orginally in `TADPOLE_D1_D2.csv`.

```

```

## Importing the datasets

The datasets can be imported by either reading in the csv or using the util.

The util will handle:
- Loading the dataset
- Selecting a featureset (specified in a configuratuion file)
- Dropping NaNs
- Splitting into an x and y, train and test sets

See the method documentation etc.

```
def load_benchmark_dataset(dataset_csv, features_yaml, dropna=False, y_column='target_diagnosis_encoded', test_size=0.1):
```

The train:test split is default 90:10 (pretty standard) and has been **shuffled** before being split.

Use the loading util as below (see `data_exploration_02.ipynb` for an example).

```
from utils import utils
import pandas as pd

dataset = './data/dataset_benchmark.csv'
featureset = './features_benchmark.yaml'

df, x, y, x_train, x_test, y_train, y_test = utils.load_benchmark_dataset(dataset, featureset, dropna=True)
```

## Featuresets

Features sets are specified in `some_set_of_features.yaml` files located in `./features/`

For example:

```
dataset: [RID]
prediction: [diagnosis_encoded, ADAS13, Ventricles]
cognitive_tests: [CDRSB, ADAS11, MMSE, RAVLT_immediate]
mri: [Hippocampus, WholeBrain, Entorhinal, MidTemp]
# pet: [FDG, AV45]
# csf: [ABETA_UPENNBIOMK9_04_19_17, TAU_UPENNBIOMK9_04_19_17, PTAU_UPENNBIOMK9_04_19_17]
risk_factors: [APOE4, AGE_AT_EXAM]
```

Commented lines are ignored.

The `load_features` util handles essentially converting a dictionary of lists into a sinlge list of features.

The example above will be converted to:

```
[RID, diagnosis_encoded, ADAS13, Ventricles, CDRSB, ADAS11, MMSE, RAVLT_immediate, Hippocampus, WholeBrain, Entorhinal, MidTemp, APOE4, AGE_AT_EXAM]
```

## **Note**

Bear in mind that datasets effectively treat each patient visit as an independent sample from some disease progression.

## Some Useful Links

- [TADPOLE](https://tadpole.grand-challenge.org/Home/)

- [TADPOLE data page](https://tadpole.grand-challenge.org/Data/)

- [Discussion of the resutls of TADPOLE](https://tadpole.grand-challenge.org/Results/)

- [TADPOLE repo](https://github.com/noxtoby/TADPOLE)

- [CMIC Summer School 2018 repo](https://github.com/mrazvan22/disProgModSummerSchool) (which itself looks like a copy of this [hackathon repo](https://github.com/swhustla/pycon2017-alzheimers-hack))

- [The missingno package repo](https://github.com/ResidentMario/missingno/blob/master/README.md)
