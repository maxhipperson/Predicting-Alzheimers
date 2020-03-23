from utils import utils
import pandas as pd
import numpy as np
import os

def main():
    
    csv_path = './data/TADPOLE_D1_D2.csv'
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
    # the csf columns should be numeric
    csf_features = ['ABETA_UPENNBIOMK9_04_19_17', 'TAU_UPENNBIOMK9_04_19_17', 'PTAU_UPENNBIOMK9_04_19_17']
    for feature in csf_features:
        df.loc[:, feature] = pd.to_numeric(df[feature], errors='coerce')
        
    # impute missing diagnosis values where the diagnosis before and after is the same
    df['prev_diagnosis'] = df.groupby('RID')['diagnosis'].shift(periods=1)
    df['next_diagnosis'] = df.groupby('RID')['diagnosis'].shift(periods=-1)
    cond = df['prev_diagnosis'] == df['next_diagnosis']
    df['diagnosis'] = df['next_diagnosis'].where(cond, df['diagnosis'])
    
    # add future predictions
    # select the prediction features and shift them up by 1 (so each visit has the prediction of the next visit)
    # --> the final visit will have NaN (because of the shift)
    # --> shift will apply per group - https://stackoverflow.com/questions/26280345/pandas-shift-down-values-by-one-row-within-a-group
    prediction_features = ['diagnosis', 'ADAS13', 'Ventricles']
    df.sort_values(['RID', 'AGE_AT_EXAM'], inplace=True)
    for feature in prediction_features:
        df.loc[:, f'target_{feature}'] = df.groupby('RID')[feature].shift(periods=-1).values
        
    # save the df
    dataset_save_path_standard = './data/dataset_standard.csv'
    df.to_csv(dataset_save_path_standard, index=False)

    ################################
    # Making the benchmark dataset #
    ################################
    
    # drop nans in diagnosis (this is the minimum we have to drop to add the target diagnosis)
    df = df.dropna(subset=['diagnosis', 'target_diagnosis'])
    
    # encode the diagnosis and target diagnosis
    from sklearn.preprocessing import OrdinalEncoder
    
    oe = OrdinalEncoder(categories=[['CN', 'MCI', 'AD']]) # map 'CN':0 'MCI':1 'AD':2
    df.loc[:, 'diagnosis_encoded'] = oe.fit_transform(df['diagnosis'].to_numpy().reshape(-1, 1)).astype(int)
    df.loc[:, 'target_diagnosis_encoded'] = oe.transform(df['target_diagnosis'].to_numpy().reshape(-1, 1)).astype(int)
    
    # For some readson there are 2s in the APOE4 column so we'll just drop these as a quick fix
    df = df[df['APOE4'] != 2.]
    
    # selecting a featureset is done through the loading util
    
    # save the df
    dataset_save_path_benchmark = './data/dataset_benchmark.csv'
    df.to_csv(dataset_save_path_benchmark, index=False)

if __name__ == "__main__":
    main()
