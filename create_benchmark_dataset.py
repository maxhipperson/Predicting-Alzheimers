import datetime
now = datetime.datetime.now()
print(f'Version: {now.strftime("%Y-%m-%d %H:%M:%S")}')

import pandas as pd
import os

def load_dataset():
    raise NotImplementedError

def preprocess_dataset():
    raise NotImplementedError

def export_dataset():
    raise NotImplementedError

# load csv
directory = './data'
filename = 'TADPOLE_D1_D2.csv'

csv_path = os.path.join(directory, filename)
df = pd.read_csv(csv_path)
df

# # add the age at exam
df.EXAMDATE = pd.to_datetime(df.EXAMDATE)
df_grouped = df.groupby('RID').apply(lambda x:(x['EXAMDATE']-x['EXAMDATE'].min()).dt.days/365.25 + x['AGE'].min())
df_grouped.sort_index(inplace=True)
df.sort_values(by=['RID','EXAMDATE'], inplace=True)
df['AGE_AT_EXAM'] = df_grouped.values
df['AGE_INT'] = df['AGE_AT_EXAM'].apply(int)
df.reset_index(drop=True)


# map the diagnosis values to those specified by the TADPOLE challenge
dx_map = {
        'MCI': 'MCI',
        'NL': 'CN',
        'Dementia': 'AD',
        'MCI to Dementia': 'AD',
        'NL to MCI': 'MCI',
        'MCI to NL': 'CN',
        'Dementia to MCI': 'MCI',
        'NL to Dementia': 'AD'
    }

df['diagnosis'] = df['DX'].map(dx_map)

# bear in mind that here we make no distrinction between catagorical and numerical fields yet

# these are the suggested biomarkers for those unfamilier with ADNI data
# according to https://tadpole.grand-challenge.org/Data/
features = {
    'dataset': ['RID', 'D1', 'D2'],
    'prediction': ['diagnosis', 'ADAS13', 'Ventricles'],
    'cognitive_tests': ['CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate'],
    'mri': ['Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp'],
    'pet': ['FDG', 'AV45'],
    'csf': ['ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17'],
    'risk_factors': ['APOE4', 'AGE_AT_EXAM']
}

# make a shortcut list of the interesting columns
columns_of_interest = []
for feature_list in features.values():
    columns_of_interest += feature_list


# The csf columns should definitely be numeric...
for feature in features['csf']:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')


# ## Making the benchmark datasets


excluded_features = features['pet'] + features['csf']
benchmark_df = df[[feature for feature in columns_of_interest if feature not in excluded_features]].dropna()
benchmark_df


# why are there 2s in the APOE4 column?


# Looks like around 10% of the APOE4 data have a value of 2...

# Just as a quick fix for the moment we'll drop these values until we figure out what's going on.


benchmark_no2s_df = benchmark_df[benchmark_df['APOE4'] != 2.].reset_index(drop=True)
benchmark_no2s_df


# add future predictions

# make list of target features
target_features = [''.join(['target_', value]) for value in features['prediction']]

# sort the data frame by RID and AGE_AT_EXAM inplace
benchmark_with_targets = benchmark_no2s_df.sort_values(['RID', 'AGE_AT_EXAM'])

grouped_by_rid = benchmark_with_targets.groupby('RID')

# select the prediction features and shift them up by 1
# (so each visit has the prediction of the next visit)
# --> the final visit will have NaN (because of the shift)
# --> shift will apply per group - https://stackoverflow.com/questions/26280345/pandas-shift-down-values-by-one-row-within-a-group
benchmark_with_targets[target_features] = grouped_by_rid[features['prediction']].shift(periods=-1)

# drop those with NaNs in the target features
benchmark_with_targets.dropna(inplace=True)
benchmark_with_targets


# seperate train and test sets based on whether participant is in D2
# we're treating each visit as a seperate sample from the disease progression

train = benchmark_with_targets[benchmark_with_targets['D2'] != 1]
test = benchmark_with_targets[benchmark_with_targets['D2'] == 1]

assert len(train) + len(test) == len(benchmark_with_targets)

print(f'train set size: {len(train)}')
print(f'test set size: {len(test)}')
print(f'full set size: {len(benchmark_with_targets)}')

# hmm... looks like we have a larger test set than training set...



# shuffle the data
import numpy as np

np.random.seed(42) # for reproducibility

shuffled_df = benchmark_with_targets.reindex(np.random.permutation(benchmark_with_targets.index)).reset_index(drop=True)
shuffled_df



# perhaps we should split the data set in an 80:20 proportion for simplicity
# assuming that each visit as a seperate sample from the disease progression

train_fraction = 0.8

train = benchmark_with_targets.iloc[:int(len(benchmark_with_targets) * train_fraction)].reset_index(drop=True)
test = benchmark_with_targets.iloc[int(len(benchmark_with_targets) * train_fraction):].reset_index(drop=True)

assert len(train) + len(test) == len(benchmark_with_targets)

print(f'train set as fraction of full set: {train_fraction}')
print(f'train set size: {len(train)}')
print(f'test set size: {len(test)}')
print(f'full set size: {len(benchmark_with_targets)}')



# save train and test sets to csv
train_csv = 'benchmark_train.csv'
test_csv = 'benchmark_test.csv'

train.to_csv(os.path.join(directory, train_csv), index=False)
test.to_csv(os.path.join(directory, test_csv), index=False)

if __name__ == "__main__":
    
    
    
    load_dataset()
    preprocess()
    create_dataset
