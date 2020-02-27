import pandas as pd
import numpy as np
import os

def main():
    
    # load csv
    directory = './data'
    
    csv = 'TADPOLE_D1_D2'
    new_csv = 'preprocessed_d1d2'
    
    csv_path = os.path.join(directory, f'{csv}.csv')
    df = pd.read_csv(csv_path)
    
    ##########################
    # Standard preprocessing #
    ##########################
    
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
    df.sort_values(by=['RID','EXAMDATE'], inplace=True)
    df.loc[:, 'AGE_AT_EXAM'] = df.groupby('RID').apply(lambda x:(x['EXAMDATE']-x['EXAMDATE'].min()).dt.days/365.25 + x['AGE'].min()).values
    
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

    ################################
    # Making the benchmark dataset #
    ################################
    
    # drop nans in diagnosis (this is the minimum we have to drop to add the target diagnosis)
    benchmark_df = df.dropna(subset=['diagnosis', 'target_diagnosis'])
    
    # select only the features we want in the benchmark dataset
    excluded_features = features['pet'] + features['csf']
    benchmark_df = benchmark_df[[feature for feature in columns_of_interest if feature not in excluded_features] + target_features]
    
    # encode the diagnosis and target diagnosis
    from sklearn.preprocessing import OrdinalEncoder
    
    oe = OrdinalEncoder(categories=[['CN', 'MCI', 'AD']]) # map 'CN':0 'MCI':1 'AD':2
    benchmark_df.loc[:, 'diagnosis_encoded'] = oe.fit_transform(benchmark_df['diagnosis'].to_numpy().reshape(-1, 1)).astype(int)
    benchmark_df.loc[:, 'target_diagnosis_encoded'] = oe.transform(benchmark_df['target_diagnosis'].to_numpy().reshape(-1, 1)).astype(int)
    
    # For some readson there are 2s in the APOE4 column so we'll just drop these as a quick fix
    benchmark_df = benchmark_df[benchmark_df['APOE4'] != 2.]
    
    ##############################
    # Save the dataframes to csv #
    ##############################
    
    new_csv_df = f'{new_csv}.csv'
    new_csv_benchmark_df = f'{new_csv}_benchmark.csv'
    
    df.to_csv(os.path.join(directory, new_csv_df), index=False)
    benchmark_df.to_csv(os.path.join(directory, new_csv_benchmark_df), index=False)

if __name__ == "__main__":
    main()
