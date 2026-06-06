import os
import numpy as np
import pandas as pd
from src.data_loader import load_config, load_batadal_data
from src.data_splitter import split_batadal_data
from src.preprocessor import TimeSeriesPreprocessor
from src.utils import create_sequences
from src.dl_models import create_lstm_model, create_gru_model, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_dl_experiments():
    print("--- DERİN ÖĞRENME (LSTM vs GRU) 5-SEED DENEYLERİ BAŞLIYOR ---")
    config = load_config("config.json")
    seeds = config['training_params']['random_seeds']
    
    # 1. Veri Hazırlığı (BATADAL)
    batadal_df = load_batadal_data(config['data_paths']['batadal'])
    train_df, val_df, test_df = split_batadal_data(batadal_df)
    
    preprocessor = TimeSeriesPreprocessor()
    exclude_cols = ['DATETIME', 'ATT_FLAG']
    train_processed = preprocessor.fit_transform(train_df, exclude_cols)
    val_processed = preprocessor.transform(val_df, exclude_cols)
    test_processed = preprocessor.transform(test_df, exclude_cols)
    
    window_size = config['automata_params']['fixed_comparison']['window_size']
    X_train, y_train = create_sequences(train_processed, 'PC1', 'ATT_FLAG', window_size)
    X_val, y_val = create_sequences(val_processed, 'PC1', 'ATT_FLAG', window_size)
    X_test, y_test = create_sequences(test_processed, 'PC1', 'ATT_FLAG', window_size)
    
    models_to_test = {
        "LSTM": create_lstm_model,
        "GRU": create_gru_model
    }
    
    all_results = []
    
    # 2. Döngü Başlıyor: Her Model ve Her Seed için
    for model_name, model_func in models_to_test.items():
        print(f"\n>>> {model_name} Modeli Test Ediliyor <<<")
        
        for seed in seeds:
            print(f"[{model_name}] Seed: {seed} ile eğitim başlıyor...")
            set_random_seed(seed)
            
            model = model_func((window_size, 1))
            early_stopping = EarlyStopping(monitor='val_loss', patience=config['training_params']['early_stopping_patience'], restore_best_weights=True)
            
            # Eğitimi sessiz yapalım (verbose=0) terminal çok dolmasın
            model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      epochs=config['training_params']['max_epochs'], 
                      batch_size=config['training_params']['batch_size'], 
                      callbacks=[early_stopping], verbose=0)
            
            # Test tahminleri
            raw_preds = model.predict(X_test, verbose=0)
            y_pred = (raw_preds > 0.5).astype(int).flatten()
            
            # Metrikler
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            all_results.append({
                "Model": model_name,
                "Seed": seed,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1_Score": f1
            })
            print(f"   -> Bitti! Acc: {acc:.4f}, F1: {f1:.4f}")

    # 3. Sonuçları Toparla ve Ortalama/Standart Sapma Hesapla
    results_df = pd.DataFrame(all_results)
    
    # Raporlama için ortalamalar ve standart sapmalar (Hocanın istediği format)
    summary_df = results_df.groupby('Model').agg(['mean', 'std']).round(4)
    
    print("\n\n=== 5 SEED DENEY SONUÇLARI (ORTALAMA ve STD) ===")
    print(summary_df)
    
    # CSV Olarak Kaydet
    os.makedirs('models', exist_ok=True)
    results_df.to_csv("models/dl_5seed_sonuclari.csv", index=False)
    summary_df.to_csv("models/dl_5seed_ozet.csv")
    print("\nTüm detaylı sonuçlar 'models' klasörüne kaydedildi!")

if __name__ == "__main__":
    run_dl_experiments()