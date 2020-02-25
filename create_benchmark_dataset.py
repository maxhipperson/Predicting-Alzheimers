import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_dataset(directory, filename):
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

    # the csf columns should be numeric
    for feature in features['csf']:
        df.loc[:, feature] = pd.to_numeric(df[feature], errors='coerce')
        
    # add future predictions
    # select the prediction features and shift them up by 1 (so each visit has the prediction of the next visit)
    # --> the final visit will have NaN (because of the shift)
    # --> shift will apply per group - https://stackoverflow.com/questions/26280345/pandas-shift-down-values-by-one-row-within-a-group
    target_features = [''.join(['target_', value]) for value in features['prediction']]
    df.sort_values(['RID', 'AGE_AT_EXAM'], inplace=True)
    for feature in features['prediction']:
        df.loc[:, f'target_{feature}'] = df.groupby('RID')[feature].shift(periods=-1).values
        
    # drop nans in diagnosis (this is the minimum we have to drop to add the target diagnosis)
    df = df.dropna(subset=['diagnosis', 'target_diagnosis'])

    ################################
    # Making the benchmark dataset #
    ################################
    
    # select only the features we want in the benchmark dataset
    excluded_features = features['pet'] + features['csf']
    benchmark_df = df[[feature for feature in columns_of_interest if feature not in excluded_features] + target_features]
    
    # encode the diagnosis and target diagnosis
    from sklearn.preprocessing import OrdinalEncoder
    
    oe = OrdinalEncoder(categories=[['CN', 'MCI', 'AD']]) # map 'CN':0 'MCI':1 'AD':2
    benchmark_df.loc[:, 'diagnosis_encoded'] = oe.fit_transform(benchmark_df['diagnosis'].to_numpy().reshape(-1, 1))
    benchmark_df.loc[:, 'target_diagnosis_encoded'] = oe.transform(benchmark_df['target_diagnosis'].to_numpy().reshape(-1, 1))
    
    # For some readson there are 2s in the APOE4 column so we'll just drop these as a quick fix
    benchmark_df = benchmark_df[benchmark_df['APOE4'] != 2.]
    
    return df, benchmark_df

def export_dataset(directory, filename, df):
    df.to_csv(os.path.join(directory, filename), index=False)

if __name__ == "__main__":
    
    import datetime
    now = datetime.datetime.now()
    print(f'Version: {now.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # load csv
    directory = './data'
    csv = 'TADPOLE_D1_D2.csv'
    
    new_csv = 'example'
    
    df_csv = f'{new_csv}.csv'
    benchmark_df_csv = f'{new_csv}_benchmark.csv'
    
    df = load_dataset(directory, csv)
    df, benchmark_df = preprocessing(df)
    
    export_dataset(directory, df_csv, df)
    export_dataset(directory, benchmark_df_csv, benchmark_df)
