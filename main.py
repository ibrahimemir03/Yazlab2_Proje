import os
from src.data_loader import load_config, load_skab_data, load_batadal_data
from src.data_splitter import split_batadal_data
from src.preprocessor import TimeSeriesPreprocessor
from src.utils import create_sequences
from src.dl_models import create_lstm_model, set_random_seed
from tensorflow.keras.callbacks import EarlyStopping
from src.evaluation import evaluate_model 

def main():
    print("1. Konfigürasyon Yükleniyor...")
    config = load_config("config.json")
    
    print("2. BATADAL Verisi Yükleniyor...")
    batadal_df = load_batadal_data(config['data_paths']['batadal'])
    
    print("3. Veri Kurallara Göre Bölünüyor (%60 Train, %20 Val, %20 Test)...")
    train_df, val_df, test_df = split_batadal_data(batadal_df)
    
    print("4. Ön İşleme (PCA & Normalizasyon) Uygulanıyor...")
    preprocessor = TimeSeriesPreprocessor()
    exclude_cols = ['DATETIME', 'ATT_FLAG']
    
    train_processed = preprocessor.fit_transform(train_df, exclude_cols)
    val_processed = preprocessor.transform(val_df, exclude_cols)
    test_processed = preprocessor.transform(test_df, exclude_cols)
    
    print("\n5. Derin Öğrenme İçin Zaman Serisi Paketleri (Sequences) Oluşturuluyor...")
    window_size = config['automata_params']['fixed_comparison']['window_size']
    
    X_train, y_train = create_sequences(train_processed, 'PC1', 'ATT_FLAG', window_size)
    X_val, y_val = create_sequences(val_processed, 'PC1', 'ATT_FLAG', window_size)
    X_test, y_test = create_sequences(test_processed, 'PC1', 'ATT_FLAG', window_size)
    
    print("\n6. LSTM Modeli Eğitiliyor (Kara Kutu Referans Modeli)...")
    set_random_seed(config['training_params']['random_seeds'][0]) 
    
    lstm_model = create_lstm_model((window_size, 1))
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=config['training_params']['early_stopping_patience'],
        restore_best_weights=True
    )
    
    history = lstm_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config['training_params']['max_epochs'],
        batch_size=config['training_params']['batch_size'],
        callbacks=[early_stopping],
        verbose=1 
    )
    
    print("\n--- LSTM EĞİTİMİ BAŞARIYLA TAMAMLANDI ---")
    
    print("\n7. LSTM Modeli Test Ediliyor...")
    # Modelden test verisi için tahminleri al (0 ile 1 arası olasılık döner)
    raw_predictions = lstm_model.predict(X_test)
    
    # Olasılıkları 0 (Normal) veya 1 (Anomali) şeklinde etiketlere çevir (0.5 sınırı ile)
    y_pred = (raw_predictions > 0.5).astype(int).flatten()
    
    # Performans metriklerini hesapla ve grafiği çizdir
    evaluate_model(y_test, y_pred, model_name="LSTM")
    # ... (LSTM test kodları burada kalıyor) ...
    
    print("\n8. OTOMATA MODELİ (Açıklanabilir Model) EĞİTİLİYOR...")
    from src.automata import ProbabilisticAutomata
    import json # JSON çıktısı için
    
    # 1. Automata'yı başlat (config.json'dan okuduğumuz parametrelerle)
    alphabet_size = config['automata_params']['fixed_comparison']['alphabet_size']
    automata = ProbabilisticAutomata(window_size=window_size, alphabet_size=alphabet_size)
    
    # 2. Modeli Train verisiyle eğit (Olasılıkları öğrensin)
    # PC1 sütununu numpy dizisi (array) olarak veriyoruz
    train_data_array = train_processed['PC1'].values
    automata.fit(train_data_array)
    
    print(f"Otomata Eğitildi. Öğrenilen Benzersiz Durum (State) Sayısı: {len(automata.known_patterns)}")
    
    print("\n9. OTOMATA MODELİ TEST EDİLİYOR & AÇIKLAMA ÜRETİLİYOR...")
    test_data_array = test_processed['PC1'].values
    
    # Raporlama için tek bir örnek üzerinden JSON çıktısı alalım (Örneğin test verisinin 5. adımı)
    time_step_to_explain = 5
    
    # Önceki durumu (Previous State) ve yeni gelen örüntüyü bulalım
    if len(test_data_array) > time_step_to_explain + window_size:
        prev_window = test_data_array[time_step_to_explain - 1 : time_step_to_explain - 1 + window_size]
        current_window = test_data_array[time_step_to_explain : time_step_to_explain + window_size]
        
        # Sayıları harflere çevir
        prev_state = automata._sax(automata._paa(prev_window))
        incoming_pattern = automata._sax(automata._paa(current_window))
        
        # Unseen kontrolü ve eşleme
        mapped_pattern, distance = automata.handle_unseen(incoming_pattern)
        status = "unseen" if distance > 0 else "seen"
        
        # Olasılık (Path Probability) hesabı
        # prev_state'den mapped_pattern'a geçiş olasılığını al, yoksa çok düşük bir ihtimal ver
        probability = automata.transition_probs.get(prev_state, {}).get(mapped_pattern, 0.001)
        
        # Karar Mekanizması: Olasılık %5'ten (0.05) düşükse Anomali de.
        decision = "anomaly" if probability < 0.05 else "normal"
        
        # Proje yönergesinde istenen ZORUNLU JSON Formatı
        explanation_output = {
            "time_step": time_step_to_explain,
            "state": prev_state,
            "pattern": incoming_pattern,
            "status": status,
            "mapped_to": mapped_pattern,
            "probability": float(f"{probability:.4f}"),
            "decision": decision
        }
        
        print("\n[ZORUNLU AÇIKLANABİLİRLİK ÇIKTISI (JSON)]")
        print(json.dumps(explanation_output, indent=4))
        
        # Açıklamayı dosyaya da kaydedelim (Hocanın görmesi için kanıt)
        with open('models/automata_explanation_sample.json', 'w') as f:
            json.dump(explanation_output, f, indent=4)
        print("JSON çıktısı 'models/automata_explanation_sample.json' olarak kaydedildi.")

if __name__ == "__main__":
    main()