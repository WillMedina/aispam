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
import joblib
'''
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
'''
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
	
#csv_path = "COLAB_GPU_dataset_spam_ham_flax-community_gpt-2-spanish_10000_48_augmented_markov2.csv"

#ruta del dataset original a pre-procesar
csv_path = 'datasets/COLAB_GPU_dataset_spam_ham_flax-community_gpt-2-spanish_10000_48_augmented_markov2.csv';
df = pd.read_csv(csv_path, dtype={'Mensaje': str})

print("====== CARGA CON EXITO DE DATOS ==================")
print(df.sample(10))
print("==================================================")
#df.head(21)

# Descargar recursos de NLTK
plt.style.use('ggplot')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

print("PRE PROCESAMIENTO ====================================== ")
# 2. Preprocesamiento inicial
def clean_text(text):
    try:
        # Verificar si es string y no está vacío
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Limpieza básica del texto
        text = text.lower()
        #text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  #los links se filtran aca
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        return text.strip()
    
    except Exception as e:
        print(f"Error limpiando texto: {str(e)} | Valor original: {text}")
        return ""

# Aplicar limpieza
# En la sección de preprocesamiento inicial, agregar:
df['mensaje'] = df['mensaje'].fillna('')  # Convertir NaNs a strings vacíos
df['mensaje'] = df['mensaje'].astype(str)  # Forzar conversión a string
df['mensaje'] = df['mensaje'].apply(clean_text)

clean_memory(True)

# Convertir variable objetivo
df['tipo_n'] = df['tipo'].apply(lambda x: 1 if x == 'spam' else 0)

# 3. Ingeniería de características
stop_words = set(stopwords.words('spanish'))  # Cambiar a 'english' si es necesario
stemmer = SnowballStemmer('spanish')  # Cambiar según idioma

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    stems = [stemmer.stem(token) for token in filtered_tokens]
    return ' '.join(stems)

clean_memory(True)
# Aplicar tokenización y stemming
df['processed_Text'] = df['mensaje'].apply(tokenize_and_stem)
clean_memory(True)

# 4. Vectorización TF-IDF
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['processed_Text']).toarray()
y = df['tipo_n'].values

# AÑADE ESTO PARA GUARDAR EL TFIDF FITTED:
output_dir = 'preprocessing_assets' # Crea este directorio si no existe
os.makedirs(output_dir, exist_ok=True) # Asegura que el directorio exista
tfidf_filename = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
joblib.dump(tfidf, tfidf_filename)
print(f"TfidfVectorizer guardado en: {tfidf_filename}")

# 5. Dividir dataset --- ESTO ES PARA EL ENTRENAMIENTO
#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=42, stratify=y
#)

clean_memory(True)
print(" DATASET DESPUES DE PREPROCESAMIENTO Y DATOS DE ENTRENAMIENTO ================ ")
print(df.head(10))
print(df.sample(20))
#print(X,y)

clean_memory(True)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
filename = f"DATASET_PREPROCESADO_G30_{timestamp}.csv"

df.to_csv(
    filename,         # Nombre del archivo
    index=False,           # No guardar el índice
    encoding='utf-8',      # Codificación para caracteres especiales
    sep=',',               # Separador de columnas
    quoting=1,             # 1=Quote solo campos necesarios, 2=Quote todos
    quotechar='"',         # Carácter para encerrar textos
    lineterminator='\n'   # Salto de línea
)

print(f"Dataset Preprocesado y transformado, guardado como: {filename}")