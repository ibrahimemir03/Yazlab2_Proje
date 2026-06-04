import numpy as np

def create_sequences(data_df, feature_col, label_col, window_size):
    """
    Zaman serisi verisini LSTM/CNN gibi modellerin anlayacağı 
    3 Boyutlu dizi (sequence) formatına dönüştürür.
    """
    features = data_df[feature_col].values
    labels = data_df[label_col].values
    
    # Detaylı Temizlik: BATADAL'daki -999 etiketlerini 0 (Normal) yapıyoruz.
    # Eğer veri setinde 1 varsa, o Anomali olarak kalır.
    labels = np.where(labels == -999, 0, labels)

    X, y = [], []
    for i in range(len(features) - window_size):
        # Pencere boyutu kadar geçmiş veriyi al
        X.append(features[i : i + window_size])
        
        # Modelin tahmin etmeye çalışacağı hedef: Bu pencereden hemen sonraki anın etiketi
        y.append(labels[i + window_size])
        
    # Modelin beklediği 3D format: (Örnek Sayısı, Zaman Adımı, Özellik Sayısı)
    # Bizim tek özelliğimiz (PC1) olduğu için son boyutu reshape(-1, window_size, 1) ile 1 yapıyoruz.
    return np.array(X).reshape(-1, window_size, 1), np.array(y)
def add_gaussian_noise(data_array, mean=0.0, std=0.05):
    """
    Deney senaryosu için veriye Gaussian gürültüsü (Gaussian Noise) ekler.
    Modelin gürültüye dayanıklılığını (Robustness) ölçmek için kullanılır.
    """
    noise = np.random.normal(mean, std, data_array.shape)
    return data_array + noise