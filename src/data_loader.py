import os
import glob
import pandas as pd
import json

def load_config(config_path="config.json"):
    """Merkezi konfigürasyon dosyasını okur."""
    with open(config_path, "r") as file:
        return json.load(file)

def load_skab_data(skab_base_path):
    """
    SKAB veri setindeki valve1 ve valve2 klasörlerindeki CSV'leri okur,
    source_group ve source_file sütunlarını ekleyerek birleştirir.
    """
    all_data = []
    target_folders = ["valve1", "valve2"]
    
    for folder in target_folders:
        folder_path = os.path.join(skab_base_path, folder)
        if not os.path.exists(folder_path):
            print(f"Uyarı: {folder_path} bulunamadı!")
            continue
            
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        
        for file_path in csv_files:
            df = pd.read_csv(file_path, sep=';', index_col=False) 
            
            # HAYAT KURTARAN KOD: Sütun adlarındaki gizli boşlukları sil
            df.columns = df.columns.str.strip()
            
            df['source_group'] = folder
            df['source_file'] = os.path.basename(file_path)
            
            all_data.append(df)
            
    if not all_data:
        raise ValueError("SKAB verileri okunamadı! Lütfen data/skab klasörünü kontrol edin.")
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def load_batadal_data(batadal_base_path):
    """BATADAL Training Dataset 2 dosyasını okur."""
    file_path = os.path.join(batadal_base_path, "BATADAL_dataset04.csv") 
    
    if not os.path.exists(file_path):
         raise ValueError(f"BATADAL verisi bulunamadı: {file_path}")
         
    df = pd.read_csv(file_path)
    
    # HAYAT KURTARAN KOD: Sütun adlarındaki gizli boşlukları sil
    df.columns = df.columns.str.strip()
    
    return df