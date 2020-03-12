# Predicting-Alzheimers

This project aims to investigate the capability of machine learning models to predict the progression of long term diseases.

We look specifically at Alzheimer's disease.

## Project Setup

Because we're only in the preliminary steps etc, all of the scipts / notebooks can just be in the top level folder.

However, it's **super important** that you make a directory called `data`.

Once you download the data (from either the [ADIS](https://ida.loni.usc.edu/login.jsp) website or the dropbox link _sent in slack_) then place `TADPOLE_D1_D2.csv` in `data`.

A.k.a. follow the directory structure below.

```
.
├── ...
├── data
│   ├── TADPOLE_D1_D2.csv
└── ...    
```

## Generating the benchmark datasets

Run `create_benchmark_dataset.py`. Make sure that `directory` is pointed at the `data` folder.

```
# load csv
directory = './data'

csv = 'TADPOLE_D1_D2'
new_csv = 'preprocessed_d1d2'
```

`csv` and `new_csv` indicate the base dataset to use and the name of the dataset ot be generated respectively.

Two datasets are created; `f'{new_csv}.csv'` and `f'{new_csv}_benchmark.csv'` which correspond to the datset with **standard preprocessing** and **benchmark preprocessing** applied respectively.

Standard preprocessing:
- Maps diagnosis
- Adds age at exam
- Adds target features
- Imputes missing diagnosis values where the diagnosis before and after is the same

Benchmark preprocessing:
- Drops NaNs in `diagnosis` and `target_diagnosis` (needed for next step)
- Selects feature subset
- Encodes `diagnosis` and `target_diagnosis`
- Drops entries where `APOE` is equal to 2

Features included in the benchmark data:

```
['RID', 'diagnosis', 'ADAS13', 'Ventricles', 'CDRSB', 'ADAS11', 'MMSE',RAVLT_immediate, 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'APOE4', 'AGE_AT_EXAM', 'target_diagnosis', 'target_ADAS13', 'target_Ventricles', 'diagnosis_encoded', 'target_diagnosis_encoded']
```

## Importing the datasets

I've made a util for importing the benchmark datasets. You can specify whether to drop entries containing NaNs, set the taget to be the `y` variable (by feature name), and specify the train:test split.

```
def load_benchmark_dataset(csv, dropna=False, y_column='target_diagnosis_encoded', test_size=0.1):
```

The train:test split is default 90:10 (pretty standard) and has been **shuffled** before being split.

Use the loading util as below (see `data_exploration_02.ipynb` for an example).

```
from utils import utils
import pandas as pd

csv = './data/preprocessed_d1d2_benchmark.csv'
df, x, y, x_train, x_test, y_train, y_test = utils.load_benchmark_dataset(csv, dropna=True)
df_na, x_na, y_na, x_train_na, x_test_na, y_train_na, y_test_na = utils.load_benchmark_dataset(csv)
```

For an example of importing the dataset with standard preproccessing see `data_exploration_01.ipynb`

## **Note**

Bear in mind that datasets effectively treat each patient visit as an independent sample from some disease progression.

## Some Useful Links

- [TADPOLE](https://tadpole.grand-challenge.org/Home/)

- [TADPOLE data page](https://tadpole.grand-challenge.org/Data/)

- [Discussion of the resutls of TADPOLE](https://tadpole.grand-challenge.org/Results/)

- [TADPOLE repo](https://github.com/noxtoby/TADPOLE)

- [CMIC Summer School 2018 repo](https://github.com/mrazvan22/disProgModSummerSchool) (which itself looks like a copy of this [hackathon repo](https://github.com/swhustla/pycon2017-alzheimers-hack))

- [The missingno package repo](https://github.com/ResidentMario/missingno/blob/master/README.md)
