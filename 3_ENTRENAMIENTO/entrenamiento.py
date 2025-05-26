#!pip install pandas scikit-learn matplotlib seaborn
#!pip install nltk tensorflow pandas scikit-learn

# -*- coding: utf-8 -*-
"""Procesamiento de datos y modelo para detección de SPAM"""
# Instalar dependencias (ejecutar solo si es necesario)
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Reshape
from tensorflow.keras.layers import AlphaDropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import save_model # PARA GUARDAR .KERAS
from datetime import datetime

import tensorflow as tf
#from google.colab import drive #solo para collab -will
import gc
import psutil 
import os

### LIMPIAR MEMORIA
def clean_memory(print_stats=False):
    """Libera memoria de forma agresiva y opcionalmente muestra estadísticas"""
    # Liberar variables no usadas
    gc.collect()
    
    # Forzar recolección de basura de objetos ciclicos
    gc.garbage.clear()
    
    if print_stats:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 ** 2)
        print(f"\nMemoria usada: {mem:.2f} MB")
    
def memory_safe_preprocess(df, processing_function, chunk_size=1000):
    """Procesamiento en chunks con limpieza de memoria"""
    clean_memory()
    
    # Dividir el dataframe en chunks
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Procesar chunk
        processed = processing_function(chunk.copy())
        processed_chunks.append(processed)
        
        # Limpiar memoria cada 5 chunks
        if i % 5 == 0:
            clean_memory()
            
        # Liberar chunk original
        del chunk
        clean_memory()
    
    return pd.concat(processed_chunks, axis=0)

# Ejemplo de uso:
def processing_function(chunk):
    # Aquí irían tus operaciones de preprocesamiento
    chunk['Mensaje'] = chunk['Mensaje'].apply(clean_text)
    return chunk

def memory_cleaner(func):
    """Decorador para limpiar memoria antes y después de una función"""
    def wrapper(*args, **kwargs):
        clean_memory()
        result = func(*args, **kwargs)
        clean_memory()
        return result
    return wrapper

def optimize_dataframe(df):
    # Reducir tamaño de tipos numéricos
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Reducir tipos float
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimizar strings
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')
    
    return df

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Uso actual de memoria: {mem:.2f} MB")
  
csv_path = 'DATASET_PREPROCESADO_G30_20250525222447.csv'; #aqui tiene que ir el csv pre-procesado
df = pd.read_csv(csv_path, dtype={'mensaje': str})
df = df.dropna();

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['processed_Text']).toarray()
y = df['tipo_n'].values

# 5. Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

'''
model_original = Sequential([
	Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_1 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_2 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5), # Un dropout un poco más alto en la primera capa puede ayudar
    Dense(128, activation='relu'),
    BatchNormalization(),
    # No Dropout aquí para probar
    Dense(1, activation='sigmoid')
])


model_3 = Sequential([
    Dense(384, activation='relu', input_shape=(X_train.shape[1],)), # Un poco menos que tu original 512
    BatchNormalization(),
    Dropout(0.5),
    Dense(192, activation='relu'), # Reducción a la mitad
    BatchNormalization(),
    Dropout(0.3),
    Dense(96, activation='relu'), # Reducción a la mitad
    Dense(1, activation='sigmoid')
])

model_4 = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)), # Mantenemos la capacidad inicial
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'), # Reducción más marcada
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),  # Capa final más pequeña antes de la salida
    Dense(1, activation='sigmoid')
])
'''

model_5 = Sequential([
    Dense(384, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(192, activation='relu'),
    BatchNormalization(),
    # No Dropout aquí para ver si la normalización es suficiente
    Dense(1, activation='sigmoid')
])

'''
model_6 = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3), # Dropout más ligero al principio
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5), # Dropout más fuerte cerca de la salida
    Dense(1, activation='sigmoid')
])

model_7 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)), # Comienza más estrecho
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'), # Se ensancha
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),  # Se vuelve a estrechar
    Dense(1, activation='sigmoid')
])

model_8 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5), # Dropout considerable
    Dense(64, activation='relu'),
    # BatchNormalization() # Podrías probar con y sin BN en esta segunda capa
    Dense(1, activation='sigmoid')
])
'''

model = model_5

clean_memory(True)

optimizer = Adam(learning_rate=0.0001)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall')
    ]
)

# 7. Entrenamiento con early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

clean_memory(True)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.3,
    callbacks=[early_stop],
    verbose=1
)

clean_memory(True)

# 8. Evaluación y métricas
def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Gráfico de precisión
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Gráfico de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['ham', 'spam'],
                yticklabels=['ham', 'spam'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión')
    plt.show()

# Generar predicciones
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Mostrar métricas
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Mostrar gráficos
plot_training_history(history)
plot_confusion_matrix(y_test, y_pred)

# 9. Evaluación final
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print(f'\nTest Accuracy: {accuracy:.4f}')
print(f'\nTest Precision: {precision:.4f}')
print(f'\nTest Recall: {recall:.4f}')
print(f'Test Loss: {loss:.4f}')

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"TALLER_G30_{timestamp}.keras"

# Guardar el modelo
save_model(model, filename)

print(f"Modelo guardado como: {filename}")