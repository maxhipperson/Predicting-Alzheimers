
def load_benchmark_dataset(csv, dropna=False, y_column='target_diagnosis_encoded', test_size=0.1):
    """Loads the bench mark dataset from csv and splits it into train and test sets.
    
    Arguments:
        csv {string} -- Path to the csv file
    
    Keyword Arguments:
        dropna {bool} -- Whether to drop NaNs from the dataset (default: {False})
        y_column {str} -- Select which target values to use (default: {'target_diagnosis_encoded'})
        test_size {float} -- Represents the proportion of the dataset to include in the test split (default: {0.1})
    
    Returns:
        Splitting -- x, y, x_train, x_test, y_train, y_test
    """
    
    import sklearn.model_selection
    import pandas as pd
    import numpy as np
    import os
    
    df = pd.read_csv(os.path.join(csv))
    
    if dropna:
        df = df.dropna()

    x_columns = ['diagnosis_encoded', 'ADAS13', 'Ventricles', 'CDRSB', 'ADAS11', 'MMSE', 'RAVLT_immediate', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'MidTemp', 'APOE4', 'AGE_AT_EXAM']

    x = df[x_columns]
    y = df[y_column]

    print(f'dataset length: {len(x)}')

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=test_size, random_state=42)

    print(f'test size: {test_size}')
    print(f'x_train length: {len(x_train)}\nx_test length: {len(x_test)}')
    
    return df, x, y, x_train, x_test, y_train, y_test
