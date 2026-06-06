import pandas as pd
from src.data_loader import load_config, load_skab_data
from src.data_splitter import get_skab_folds
from src.preprocessor import TimeSeriesPreprocessor
from src.utils import create_sequences
from src.dl_models import create_lstm_model, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

def run_skab_experiments():
    print("--- SKAB VERİ SETİ (GROUP-KFOLD) DENEYİ BAŞLIYOR ---")
    config = load_config("config.json")
    
    # 1. SKAB Verisini Yükle
    skab_df = load_skab_data(config['data_paths']['skab'])
    
    # 2. GroupKFold ile böl
    folds = get_skab_folds(skab_df, n_splits=5)
    
    fold_accuracies = []
    
    for fold_num, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n>>> FOLD {fold_num} TEST EDİLİYOR <<<")
        
        train_df = skab_df.iloc[train_idx].copy()
        test_df = skab_df.iloc[test_idx].copy()
        
        # SKAB için exclude_cols
        exclude_cols = ['datetime', 'changepoint', 'anomaly', 'source_group', 'source_file']
        
        preprocessor = TimeSeriesPreprocessor()
        train_processed = preprocessor.fit_transform(train_df, exclude_cols)
        test_processed = preprocessor.transform(test_df, exclude_cols)
        
        window_size = config['automata_params']['fixed_comparison']['window_size']
        
        X_train, y_train = create_sequences(train_processed, 'PC1', 'anomaly', window_size)
        X_test, y_test = create_sequences(test_processed, 'PC1', 'anomaly', window_size)
        
        set_random_seed(42)
        model = create_lstm_model((window_size, 1))
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32, callbacks=[early_stopping], verbose=0)
        
        raw_preds = model.predict(X_test, verbose=0)
        y_pred = (raw_preds > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        print(f"Fold {fold_num} Accuracy: {acc:.4f}")

    print(f"\n=== SKAB GROUP-KFOLD SONUCU ===")
    print(f"Ortalama Accuracy: {sum(fold_accuracies)/len(fold_accuracies):.4f}")

if __name__ == "__main__":
    run_skab_experiments()