import pandas as pd
from src.data_loader import load_config, load_batadal_data
from src.data_splitter import split_batadal_data
from src.preprocessor import TimeSeriesPreprocessor
from src.automata import ProbabilisticAutomata
from src.utils import add_gaussian_noise

def run_automata_parameter_analysis():
    print("--- OTOMATA PARAMETRE DUYARLILIK ANALİZİ BAŞLIYOR ---")
    config = load_config("config.json")
    
    # 1. Veriyi Hızlıca Hazırla
    batadal_df = load_batadal_data(config['data_paths']['batadal'])
    train_df, _, _ = split_batadal_data(batadal_df)
    
    preprocessor = TimeSeriesPreprocessor()
    
    # HATA BURADAYDI, DÜZELTİLDİ: Doğrudan liste olarak veriyoruz.
    train_processed = preprocessor.fit_transform(train_df, ['DATETIME', 'ATT_FLAG'])
    train_data_array = train_processed['PC1'].values
    
    # 2. Config dosyasındaki parametre listelerini çek
    window_sizes = config['automata_params']['variations']['window_sizes']
    alphabet_sizes = config['automata_params']['variations']['alphabet_sizes']
    
    results = []
    
    # 3. Tüm kombinasyonları test et
    for w in window_sizes:
        for a in alphabet_sizes:
            automata = ProbabilisticAutomata(window_size=w, alphabet_size=a)
            automata.fit(train_data_array)
            state_count = len(automata.known_patterns)
            
            results.append({
                "Window Size": w,
                "Alphabet Size": a,
                "Öğrenilen State Sayısı": state_count
            })
            print(f"Test Ediliyor -> Window: {w}, Alphabet: {a} | State Sayısı: {state_count}")
            
    # Sonuçları güzel bir tabloya çevirip kaydet
    results_df = pd.DataFrame(results)
    print("\n--- DENEY SONUÇ TABLOSU ---")
    print(results_df.to_string(index=False))
    
    results_df.to_csv("models/parametre_analizi_sonuclari.csv", index=False)
    print("\nBu sonuçlar 'models/parametre_analizi_sonuclari.csv' dosyasına kaydedildi. Raporuna direkt ekleyebilirsin!")

if __name__ == "__main__":
    run_automata_parameter_analysis()