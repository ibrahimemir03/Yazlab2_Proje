import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_parameter_analysis():
    print("Otomata Parametre Grafiği Çiziliyor...")
    
    # CSV'yi oku
    try:
        df = pd.read_csv("models/parametre_analizi_sonuclari.csv")
    except FileNotFoundError:
        print("Hata: parametre_analizi_sonuclari.csv bulunamadı.")
        return

    plt.figure(figsize=(10, 6))
    
    # Seaborn ile çizgi grafiği
    sns.lineplot(data=df, x='Window Size', y='Öğrenilen State Sayısı', hue='Alphabet Size', marker='o')
    
    plt.title('Automata Parametre Duyarlılık Analizi')
    plt.xlabel('Window Size (Pencere Boyutu)')
    plt.ylabel('Öğrenilen Benzersiz Durum (State) Sayısı')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Alphabet Size')
    
    # Kaydet
    os.makedirs('models', exist_ok=True)
    save_path = "models/parametre_duyarlilik_grafiki.png"
    plt.savefig(save_path)
    print(f"Grafik '{save_path}' konumuna kaydedildi!")

if __name__ == "__main__":
    plot_parameter_analysis()