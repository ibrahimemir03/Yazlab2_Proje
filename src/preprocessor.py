import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

class TimeSeriesPreprocessor:
    def __init__(self):
        # Normalizasyon (0-1 aralığı) ve PCA nesnelerini başlatıyoruz
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=1)
        self.is_fitted = False
        
    def fit_transform(self, train_data, columns_to_exclude):
        """
        SADECE Eğitim verisi üzerinde kuralları öğrenir (fit) ve uygular (transform).
        Data Leakage (veri sızıntısı) olmaması için kritik fonksiyondur.
        """
        # Modele verilmemesi gereken sütunları ayır
        features = train_data.drop(columns=columns_to_exclude, errors='ignore')
        
        # 1. Normalizasyon (Ölçeklendirme)
        scaled_features = self.scaler.fit_transform(features)
        
        # 2. PCA ile Tek Boyuta (PC1) İndirgeme
        pc1_features = self.pca.fit_transform(scaled_features)
        
        # Sonuçları DataFrame'e çevir
        result_df = pd.DataFrame(pc1_features, columns=['PC1'], index=train_data.index)
        
        # Orijinal veriden silmediğimiz (zaman, etiket gibi) sütunları geri ekle
        for col in columns_to_exclude:
            if col in train_data.columns:
                result_df[col] = train_data[col]
                
        self.is_fitted = True
        return result_df

    def transform(self, test_data, columns_to_exclude):
        """
        Test veya Doğrulama (Validation) verisine, EĞİTİMDE ÖĞRENİLEN kuralları uygular.
        Burada kesinlikle 'fit' kullanılmaz!
        """
        if not self.is_fitted:
            raise ValueError("Önce fit_transform fonksiyonu çağrılmalıdır!")
            
        features = test_data.drop(columns=columns_to_exclude, errors='ignore')
        
        # Eğitimden öğrenilen kurallarla uygula
        scaled_features = self.scaler.transform(features)
        pc1_features = self.pca.transform(scaled_features)
        
        result_df = pd.DataFrame(pc1_features, columns=['PC1'], index=test_data.index)
        
        for col in columns_to_exclude:
            if col in test_data.columns:
                result_df[col] = test_data[col]
                
        return result_df