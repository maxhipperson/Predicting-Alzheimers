import yaml

def load_features(filepath):
    """Loads a featureset from a yaml file.
    
    Arguments:
        filepath {string} -- Path to the file containing the featureset.
    
    Returns:
        List -- List of features
    """
    
    print(f'Loading features from {filepath}')
    
    with open(filepath) as file:
        features_yaml = yaml.safe_load(file)
    features = [feature for featureset in features_yaml.values() for feature in featureset]
    
    return features

def load_benchmark_dataset(dataset_csv, features_yaml, dropna=False, y_column='target_diagnosis_encoded', test_size=0.1):
    """Loads the bench mark dataset from csv and splits it into train and test sets.
    
    Arguments:
        dataset_csv {string} -- Path to the csv file containing the dataset
        features_yaml {string} -- Path to the yaml file containing the feature set
    
    Keyword Arguments:
        dropna {bool} -- Whether to drop NaNs from the dataset (default: {False})
        y_column {str} -- Select which target values to use (default: {'target_diagnosis_encoded'})
        test_size {float} -- Represents the proportion of the dataset to include in the test split (default: {0.1})
    
    Returns:
        Splitting -- df, x, y, x_train, x_test, y_train, y_test
    """
    
    import sklearn.model_selection
    import pandas as pd
    import numpy as np
    import os
    
    print(f'Loading dataset from {dataset_csv}')
    
    df = pd.read_csv(dataset_csv)

    x_columns = load_features(features_yaml)
    df = df[x_columns + [y_column]]
    
    if dropna:
        df = df.dropna()
        
    x = df[x_columns]
    y = df[y_column]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=42)
    print(f'dataset length: {len(df)}')
    print(f'x_train length: {len(x_train)}\ny_train length: {len(y_train)}')
    print(f'x_test length: {len(x_test)}\ny_test length: {len(y_test)}')
    
    return df, x, y, x_train, x_test, y_train, y_test
