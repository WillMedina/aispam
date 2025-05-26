import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # Asegúrate que nltk y sus datos estén disponibles

from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib # Para cargar el TfidfVectorizer
import pandas as pd # Para el pd.isna si decides usarlo

# --- Configuración ---
app = Flask(__name__)
MODEL_PATH = 'TALLER_G30_20250525222945.keras' # Cambia al nombre de tu archivo .keras
TFIDF_VECTORIZER_PATH = 'preprocessing_assets/tfidf_vectorizer.pkl'

# --- Descargar recursos de NLTK (importante para el entorno de Docker) ---
# Esto se puede ejecutar una vez al iniciar la app o mejor aún, en el Dockerfile.
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Cargar Modelo y TfidfVectorizer ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Modelo {MODEL_PATH} cargado exitosamente.")
    
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    print(f"TfidfVectorizer {TFIDF_VECTORIZER_PATH} cargado exitosamente.")

except Exception as e:
    print(f"Error al cargar el modelo o TfidfVectorizer: {e}")
    model = None
    tfidf_vectorizer = None

# --- Funciones de Preprocesamiento (copiadas/adaptadas de tu script) ---
stop_words_spanish = set(stopwords.words('spanish'))
stemmer_spanish = SnowballStemmer('spanish')

def clean_text(text):
    try:
        if not isinstance(text, str) or pd.isna(text): # pd.isna es más robusto que solo text == ""
            return ""
        text = text.lower()
        # text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Descomenta si lo usaste
        text = re.sub(r'@\w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text) # Quita puntuación etc.
        text = re.sub(r'\d+', '', text)     # Quita números
        return text.strip()
    except Exception as e:
        print(f"Error limpiando texto: {str(e)} | Valor original: {text}")
        return ""

def tokenize_and_stem(text):
    if not text: # Si clean_text devolvió ""
        return ""
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words_spanish]
    stems = [stemmer_spanish.stem(token) for token in filtered_tokens]
    return ' '.join(stems)

def preprocess_input_text(text_message):
    """
    Pipeline completo de preprocesamiento para un solo mensaje.
    """
    if tfidf_vectorizer is None:
        raise ValueError("TfidfVectorizer no está cargado.")
        
    cleaned_message = clean_text(text_message)
    processed_message_text = tokenize_and_stem(cleaned_message)
    
    # TF-IDF espera una lista de documentos (strings)
    # Transformamos y convertimos a array denso como en el entrenamiento
    vectorized_message = tfidf_vectorizer.transform([processed_message_text]).toarray()
    
    # Asegurarse de que la forma sea (1, num_features) que es lo que espera el modelo
    # num_features debe ser igual a X_train.shape[1] (ej. 5000 en tu caso)
    # Esto ya debería ser manejado por el tfidf_vectorizer.transform
    print(f"Forma del vector TF-IDF: {vectorized_message.shape}") # Para depuración
    return vectorized_message

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tfidf_vectorizer is None:
        return jsonify({'error': 'Modelo o Vectorizador no cargado'}), 500

    try:
        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'Falta el campo "message"'}), 400

        message_text = data['message']
        
        # Preprocesar el mensaje de entrada
        processed_input_vector = preprocess_input_text(message_text)
        
        # Realizar la predicción
        prediction_proba = model.predict(processed_input_vector)[0][0]
        
        threshold = 0.5 # Umbral de decisión
        binary_prediction = 1 if prediction_proba >= threshold else 0
        label = "spam" if binary_prediction == 1 else "ham"

        return jsonify({
            'original_message': message_text,
            'processed_for_tfidf': tokenize_and_stem(clean_text(message_text)), # Para ver qué entró al TFIDF
            'prediction_label': label,
            'prediction_probability_spam': float(prediction_proba),
            'is_spam': binary_prediction
        })

    except ValueError as ve: # Errores específicos como vectorizador no cargado
        print(f"Error de valor en /predict: {ve}")
        return jsonify({'error': str(ve)}), 500
    except Exception as e:
        # Considera loggear el traceback completo para depuración:
        # import traceback
        # print(f"Error en /predict: {e}\n{traceback.format_exc()}")
        print(f"Error inesperado en /predict: {e}")
        return jsonify({'error': f'Error procesando la solicitud: {str(e)}'}), 500

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True) # Para desarrollo local
    # Para producción (Docker), Gunicorn o Waitress es mejor (ver Fase 2 de la respuesta anterior).
    # El CMD en el Dockerfile ya usa Gunicorn.
    # Si quieres probar con Gunicorn localmente (después de `pip install gunicorn`):
    # gunicorn --bind 0.0.0.0:5000 app:app
    print("Iniciando servidor Flask. Usar Gunicorn en producción.")
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False cuando uses Gunicorn