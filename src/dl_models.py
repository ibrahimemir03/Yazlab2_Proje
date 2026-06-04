import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Dense, Dropout, GlobalAveragePooling1D

def set_random_seed(seed_value):
    """
    Proje yönergesinde istenen rastgelelik kontrolü.
    Deneylerin tekrar edilebilir olması için zorunludur.
    """
    tf.random.set_seed(seed_value)

def create_lstm_model(input_shape):
    """LSTM tabanlı derin öğrenme modeli."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        # İkili sınıflandırma (Anomali var/yok) için sigmoid
        Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_gru_model(input_shape):
    """GRU tabanlı derin öğrenme modeli."""
    model = Sequential([
        GRU(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape):
    """1D-CNN tabanlı derin öğrenme modeli."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model