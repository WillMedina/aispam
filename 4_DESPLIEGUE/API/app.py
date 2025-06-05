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
from datetime import datetime # Para la fecha de revisión

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
    '''
    if not text: # Si clean_text devolvió ""
        return ""
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words_spanish]
    stems = [stemmer_spanish.stem(token) for token in filtered_tokens]
    return ' '.join(stems)
    '''
    if not text:
        return ""
    try:
        tokens = nltk.word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in stop_words_spanish]
        stems = [stemmer_spanish.stem(token) for token in filtered_tokens]
        return ' '.join(stems)
    except Exception as e:
        print(f"Error tokenize_and_stem: {str(e)} | Valor original: {text}")
        return ""

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

# --- NUEVO ENDPOINT: /importcsv ---
@app.route('/importcsv', methods=['POST'])
def import_csv():
    if model is None or tfidf_vectorizer is None:
        return jsonify({'error': 'Servicio no disponible: Modelo o Vectorizador no cargado'}), 503

    try:
        json_data = request.get_json()
        
        # Validación básica de la estructura esperada (lista con un elemento)
        if not isinstance(json_data, list) or len(json_data) == 0:
            return jsonify({'error': 'JSON de entrada debe ser una lista con al menos un objeto CSV'}), 400
        
        csv_object = json_data[0] # Asumimos que siempre viene un solo "CSV" en la lista

        if 'metadata' not in csv_object or 'registros' not in csv_object:
            return jsonify({'error': 'Objeto CSV debe contener "metadata" y "registros"'}), 400

        metadata = csv_object['metadata']
        registros = csv_object['registros']

        if not isinstance(registros, list):
            return jsonify({'error': '"registros" debe ser una lista'}), 400

        # Para procesamiento por lotes más eficiente:
        # 1. Recolectar todos los mensajes
        # 2. Preprocesarlos todos
        # 3. Vectorizarlos todos
        # 4. Predecir todos
        
        mensajes_originales = [reg.get('mensaje', '') for reg in registros]
        
        # Preprocesamiento en lote
        processed_texts_for_tfidf = []
        for msg in mensajes_originales:
            cleaned = clean_text(msg)
            processed_texts_for_tfidf.append(tokenize_and_stem(cleaned))

        # Vectorización en lote
        # Manejar el caso de que todos los mensajes procesados resulten vacíos
        if not any(processed_texts_for_tfidf): # Si todos son strings vacíos
            # Crear vectores de ceros o la representación que tu modelo espera para "nada"
            # El transform de TFIDF con "" da un vector de ceros si el vocabulario está vacío o no hay matches
            # Si el vectorizador está fitteado, dará la representación de un string vacío.
            # Esto debería ser manejado correctamente por el transform si se le da una lista de strings vacíos.
            if not processed_texts_for_tfidf: # lista de strings vacíos si todos los mensajes eran vacíos.
                vectorized_batch = tfidf_vectorizer.transform(["" for _ in range(len(registros))]).toarray()
            else: # Si algunos son vacíos y otros no.
                 vectorized_batch = tfidf_vectorizer.transform(processed_texts_for_tfidf).toarray()

        else:
            vectorized_batch = tfidf_vectorizer.transform(processed_texts_for_tfidf).toarray()
        
        # Predicción en lote
        predictions_probas_batch = model.predict(vectorized_batch)

        # Actualizar cada registro
        current_timestamp = datetime.now().isoformat()
        model_name_used = os.path.basename(MODEL_PATH) # Obtiene solo el nombre del archivo

        for i, reg in enumerate(registros):
            reg['fecha_revision'] = current_timestamp
            reg['probabilidad_spam'] = float(predictions_probas_batch[i][0])
            # Para 'tfidf_generado', guardaremos el texto que entró al vectorizador TF-IDF.
            # Guardar el vector numérico completo en JSON es poco práctico y muy verboso.
            reg['tfidf_generado'] = processed_texts_for_tfidf[i]
            reg['modelo_usado'] = model_name_used
            # La variable 'tipo_recibido' se mantiene tal como llegó.

        # Actualizar metadata
        metadata['procesado'] = True
        metadata['fecha_procesamiento_api'] = current_timestamp # Añadir fecha de procesamiento

        # Devolver la estructura JSON modificada
        return jsonify(json_data) # json_data ya está modificado y es una lista

    except Exception as e:
        print(f"Error en /importcsv: {e}") # Loggear el error
        # import traceback
        # print(traceback.format_exc()) # Para debug más detallado
        return jsonify({'error': f'Error procesando la solicitud CSV: {str(e)}'}), 500

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True) # Para desarrollo local
    # Para producción (Docker), Gunicorn o Waitress es mejor (ver Fase 2 de la respuesta anterior).
    # El CMD en el Dockerfile ya usa Gunicorn.
    # Si quieres probar con Gunicorn localmente (después de `pip install gunicorn`):
    # gunicorn --bind 0.0.0.0:5000 app:app
    print("Iniciando servidor Flask. Usar Gunicorn en producción.")
    app.run(host='0.0.0.0', port=5000, debug=False) # debug=False cuando uses Gunicorn