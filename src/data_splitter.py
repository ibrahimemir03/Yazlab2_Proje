import pandas as pd
from sklearn.model_selection import GroupKFold

def split_batadal_data(df):
    """
    BATADAL verisini zaman sırasını koruyarak tam olarak 
    %60 Train, %20 Validation, %20 Test olarak böler.
    """
    total_rows = len(df)
    train_end = int(total_rows * 0.60)
    val_end = int(total_rows * 0.80)
    
    train_data = df.iloc[:train_end].copy()
    val_data = df.iloc[train_end:val_end].copy()
    test_data = df.iloc[val_end:].copy()
    
    return train_data, val_data, test_data

def get_skab_folds(df, n_splits=5):
    """
    SKAB verisini source_file bazlı GroupKFold ile 5 parçaya böler.
    Aynı dosyadan gelen verilerin train ve test kümelerine sızmasını engeller.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df['source_file'].values
    
    folds = []
    # Train ve Test indekslerini ayırıyoruz
    for train_idx, test_idx in gkf.split(df, groups=groups):
        folds.append((train_idx, test_idx))
        
    return folds