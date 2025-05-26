# 1. Usar una imagen base de Python oficial
FROM python:3.9-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar el archivo de requerimientos e instalar dependencias primero
# Esto aprovecha el caché de Docker si los requerimientos no cambian
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Descargar recursos de NLTK necesarios
# Esto asegura que estén disponibles en la imagen
RUN python -m nltk.downloader stopwords punkt punkt_tab

# 5. Copiar el resto del código de la aplicación al directorio de trabajo
COPY . .
# Específicamente, esto copiará:
# - app.py
# - tu_modelo.keras (asegúrate que este nombre coincida)
# - la carpeta preprocessing_assets/ con tfidf_vectorizer.pkl

# 6. Exponer el puerto en el que Gunicorn servirá la aplicación
EXPOSE 5000

# 7. Comando para ejecutar la aplicación Flask usando Gunicorn
# Gunicorn es un servidor WSGI robusto para producción.
# app:app se refiere al objeto 'app' dentro del archivo 'app.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]