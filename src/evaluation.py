import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Modelin performansını istenen metriklerle hesaplar ve ekrana yazdırır.
    Ayrıca Confusion Matrix görselini 'models' klasörüne kaydeder.
    """
    accuracy = accuracy_score(y_true, y_pred)
    # Sıfıra bölme hatalarını engellemek için zero_division=0 ekliyoruz
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"\n--- {model_name} TEST SONUÇLARI ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("-" * 30)
    
    # Confusion Matrix Çizimi
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Tahmin Edilen (Predicted)')
    plt.ylabel('Gerçek (Actual)')
    
    # Görseli kaydet (models klasörü içine)
    os.makedirs('models', exist_ok=True)
    save_path = f'models/{model_name.lower()}_confusion_matrix.png'
    plt.savefig(save_path)
    plt.close()
    
    print(f"[{model_name}] Confusion matrix grafiği '{save_path}' konumuna kaydedildi.")
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}