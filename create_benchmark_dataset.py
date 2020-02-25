import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_dataset(directory, filename):
    
    # load csv
    csv_path = os.path.join(directory, filename)
    df = pd.read_csv(csv_path)
    return df

def preprocessing(df):
    
    # --> tadpole standard preprocessing
    
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

    df.loc[:, 'diagnosis'] = df['DX'].map(dx_map)
    
    # add the age at exam
    df.EXAMDATE = pd.to_datetime(df.EXAMDATE)
    df_grouped = df.groupby('RID').apply(lambda x:(x['EXAMDATE']-x['EXAMDATE'].min()).dt.days/365.25 + x['AGE'].min())
    df_grouped.sort_index(inplace=True)
    df.sort_values(by=['RID','EXAMDATE'], inplace=True)
    df.loc[:, 'AGE_AT_EXAM'] = df_grouped.values
    # df['AGE_INT'] = df['AGE_AT_EXAM'].apply(int)
    df.reset_index(drop=True)
    
    # bear in mind that here we make no distrinction between catagorical and numerical fields yet

    # these are the suggested biomarkers for those unfamilier with ADNI data
    # according to https://tadpole.grand-challenge.org/Data/
    features = {
        'dataset': ['RID'],
        # 'dataset': ['RID', 'D1', 'D2'],
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

    # The csf columns should be numeric
    for feature in features['csf']:
        df.loc[:, feature] = pd.to_numeric(df[feature], errors='coerce')
        
    # --> add future predictions
    
    # sort the data frame by RID and AGE_AT_EXAM inplace
    
    # select the prediction features and shift them up by 1
    # (so each visit has the prediction of the next visit)
    # --> the final visit will have NaN (because of the shift)
    # --> shift will apply per group - https://stackoverflow.com/questions/26280345/pandas-shift-down-values-by-one-row-within-a-group
    target_features = [''.join(['target_', value]) for value in features['prediction']]
    df.sort_values(['RID', 'AGE_AT_EXAM'], inplace=True)
    # grouped_by_rid = df.groupby('RID')
    for feature in features['prediction']:
        df.loc[:, f'target_{feature}'] = df.groupby('RID')[feature].shift(periods=-1).values

    # --> Making the benchmark datasets
    
    # exclude features
    excluded_features = features['pet'] + features['csf']
    benchmark_df = df[[feature for feature in columns_of_interest if feature not in excluded_features] + target_features]
    
    # drop nans
    benchmark_df = benchmark_df.dropna(subset=features['prediction'] + target_features)
    
    from sklearn.preprocessing import OrdinalEncoder
    
    # encode the diagnosis and target diagnosis
    oe = OrdinalEncoder(categories=[['CN', 'MCI', 'AD']]) # map 'CN':0 'MCI':1 'AD':2 as categories
    benchmark_df.loc[:, 'diagnosis_encoded'] = oe.fit_transform(benchmark_df['diagnosis'].to_numpy().reshape(-1, 1))
    benchmark_df.loc[:, 'target_diagnosis_encoded'] = oe.transform(benchmark_df['target_diagnosis'].to_numpy().reshape(-1, 1))
    
    # why are there 2s in the APOE4 column?
    # Looks like around 10% of the APOE4 data have a value of 2...
    # Just as a quick fix for the moment we'll drop these values until we figure out what's going on.
    benchmark_no2s_df = benchmark_df[benchmark_df['APOE4'] != 2.]
    
    return benchmark_no2s_df

def export_dataset(directory, filename, df):
    
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    
    import datetime
    now = datetime.datetime.now()
    print(f'Version: {now.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # load csv
    directory = './data'
    csv = 'TADPOLE_D1_D2.csv'
    new_csv = 'example.csv'
    
    df = load_dataset(directory, csv)
    df = preprocessing(df)
    export_dataset(directory, new_csv, df)
