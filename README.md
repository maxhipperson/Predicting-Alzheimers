# Predicting-Alzheimers

This project aims to investigate the capability of machine learning models to predict the progression of long term diseases.

We look specifically at Alzheimer's disease.

## Project Setup

Because we're only in the preliminary steps etc, all of the scipts / notebooks can just be in the top level folder.

However, it's **super important** that you make a directory called `data`.

Once you download the data (from either the [ADIS](https://ida.loni.usc.edu/login.jsp) webiste or the dropbox link _sent in slack_) then place `TADPOLE_D1_D2.csv` in `data`.

A.k.a. follow the directory structure below.

```
.
├── ...
├── data
│   ├── benchmark_test.csv
│   ├── benchmark_train.csv
│   ├── TADPOLE_D1_D2.csv
└── ...    
```

## Generating the benchmark datasets

The benchmark data sets should be in the dropbox folder, but if you want to change / regenerate them then open  `preliminary_data_exploration.ipynb` and run all the cells.

You might have to be careful about running the cells using the `missingno` package but the rest is jsut standard `numpy` and `pandas`.

### Benchmark dataset info

Features included in the benchmark data:

```
['D1', 'D2', 'diagnosis', 'ADAS13', 'Ventricles', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'APOE4', 'AGE_AT_EXAM']
```

The train:test split is currently 80:20 (pretty standard), and has been **shuffled** before being split.

This can be changed in the notebook by changing the value of `train_fraction` which sets the value of the fraction of the benchmark set to use for the train data (pretty self explanatory...).

The `test_fraction` is automatically calculated.

Bear in mind that this essentially ignores repeat patient visits, and treats each visit as an independent sample from some disease progression.

Just read the csvs in with `pandas.read_csv(path/to/csv)` (I'm sure you guys know how to do it).

## Some Useful Links

- [TADPOLE](https://tadpole.grand-challenge.org/Home/)

- [TADPOLE data page](https://tadpole.grand-challenge.org/Data/)

- [Discussion of the resutls of TADPOLE](https://tadpole.grand-challenge.org/Results/)

- [TADPOLE repo](https://github.com/noxtoby/TADPOLE)

- [CMIC Summer School 2018 repo](https://github.com/mrazvan22/disProgModSummerSchool) (which itself looks like a copy of this [hackathon repo](https://github.com/swhustla/pycon2017-alzheimers-hack))

- [The missingno package repo](https://github.com/ResidentMario/missingno/blob/master/README.md)
