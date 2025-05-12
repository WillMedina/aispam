'''
Módulo para generar mensajes spam y ham usando modelos de texto en español y estructurar un dataset balanceado.
Compatible con entornos locales y Colab; detecta GPU y usa CPU si no está disponible.
'''
# Instalar dependencias (en Colab)
#!pip install pandas transformers torch #--quiet
import os
import random
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM #, BitsAndBytesConfig # Opcional para cuantización
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import time # Para medir tiempo
import unicodedata # Necesario para normalización y categorías Unicode

# ------------------------------------------------------
# Configuración
# ------------------------------------------------------
# print(f"Dispositivo detectado inicialmente: {DEVICE}")
# if DEVICE != 'cuda':
# print("AVISO: No se detecta GPU. Se usará CPU, lo cual será más lento.")

# --- Modelos a considerar ---
# Buenas opciones generales (equilibrio velocidad/calidad):
# MODEL_NAME = 'datificate/gpt2-small-spanish'
# MODEL_NAME = 'DeepESP/gpt2-spanish'
MODEL_NAME = 'flax-community/gpt-2-spanish' # Tu elección actual, buena
#MODEL_NAME = 'mrm8488/spanish-gpt2'

# Opciones más grandes/potentes (requieren más VRAM/tiempo):
# MODEL_NAME = 'bigscience/bloom-560m'
# MODEL_NAME = 'somosnlp/gpt2-medium-spanish' # Un poco más grande que gpt2-small

# Modelos muy grandes (probablemente necesiten cuantización o GPUs potentes):
# MODEL_NAME = 'ablanchar/Mistral-7B-Instruct-v0.1-OA-ES' # Muy bueno, pero grande

# --- Configuración de Cuantización (Opcional, para modelos grandes o si hay OOM en GPU) ---
# USE_QUANTIZATION = False # Poner a True para probar
# if USE_QUANTIZATION and torch.cuda.is_available():
# try:
#         quantization_config = BitsAndBytesConfig(
# load_in_8bit=True,
# # bnb_4bit_compute_dtype=torch.float16 # Opcional para 4bit
#         )
# print("Usando cuantización de 8 bits con BitsAndBytes.")
# except ImportError:
# print("BitsAndBytes no instalado. No se usará cuantización. pip install bitsandbytes")
#         USE_QUANTIZATION = False
#         quantization_config = None
# else:
#     USE_QUANTIZATION = False
#     quantization_config = None

# --- Carga de modelo y tokenizer ---
# Mover la detección de dispositivo y carga aquí para que `device_map` funcione bien con cuantización
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Asegurar que el pad_token es eos_token si no está definido
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Cargando modelo {MODEL_NAME}...")

# Configuración de dispositivo y carga de modelo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo seleccionado: {DEVICE}")

if DEVICE == 'cuda':
    try:
        # if USE_QUANTIZATION and quantization_config:
        # model = AutoModelForCausalLM.from_pretrained(
        # MODEL_NAME,
        # quantization_config=quantization_config,
        # device_map="auto" # BitsAndBytes maneja la distribución a GPU
        # )
        # print(f"Modelo {MODEL_NAME} cargado en GPU con cuantización (si está habilitada y es compatible).")
        # else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        print(f"Modelo {MODEL_NAME} cargado en {DEVICE}.")
        
        # torch.compile es más efectivo en versiones recientes de PyTorch y con GPUs compatibles
        print("Compilando modelo con torch.compile (puede tardar unos minutos la primera vez)...")
        model = torch.compile(model, mode="reduce-overhead") # mode="max-autotune" para más optimización pero más tiempo de compilación
        print("Modelo compilado.")
        
    except Exception as e:
        print(f"Error al cargar o compilar modelo en GPU: {e}. Intentando en CPU.")
        DEVICE = 'cpu' # Forzar CPU si hay error en GPU
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
        print(f"Modelo {MODEL_NAME} cargado en CPU.")
        
else:
    print("AVISO: No se detecta GPU o se forzó CPU. Se usará CPU, lo cual será más lento.")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # model_cpu_quantized = torch.quantization.quantize_dynamic( # Ejemplo cuantización CPU
    # model.to('cpu'), {torch.nn.Linear}, dtype=torch.qint8
    # )
    # model = model_cpu_quantized
    print(f"Modelo {MODEL_NAME} cargado en CPU.")


model.eval() # Importante para la inferencia

# ------------------------------------------------------
# Limpieza de mensajes
# ------------------------------------------------------
def clean_text(text_input: str) -> str:
    if not isinstance(text_input, str):
        return ""

    text = text_input

    # 1. Normalización Unicode a NFC (Forma de Composición Normalizada)
    # Ayuda a tener una representación canónica de caracteres (ej. acentos)
    try:
        text = unicodedata.normalize('NFC', text)
    except TypeError: # Por si acaso llega algo que no es exactamente un string tras una conversión
        return ""

    # 2. Eliminar Emojis y muchos otros símbolos (usando rangos Unicode)
    # Esta es una aproximación. Para una solución más completa, considerar la librería 'emoji'.
    # pip install emoji
    # import emoji
    # text = emoji.replace_emoji(text, replace=' ')
    try:
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002700-\U000027BF"  # Dingbats
            "\U00002B50"             # Star emoji (ejemplo individual)
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' ', text) # Reemplazar por espacio
        
    except re.error: # Por si la regex es demasiado compleja en alguna plataforma de Python antigua
        print("Advertencia: No se pudo compilar la regex de emojis compleja. Usando un filtro más simple.")
        # Un filtro mucho más simple (menos efectivo para emojis, pero más seguro)
        text = ''.join(char for char in text if unicodedata.category(char)[0] not in ['So']) # So = Symbol, other

    # 3. Reemplazar saltos de línea
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 4. Reemplazar comillas problemáticas
    text = text.replace('"', "'") # Doble estándar a simple
    text = text.replace('“', "'").replace('”', "'") # Dobles tipográficas a simple
    text = text.replace('‘', "'").replace('’', "'") # Simples tipográficas a simple estándar
    
    # 5. Filtrar para mantener solo caracteres "deseables"
    # Permitiremos letras, números, espacios y un conjunto de puntuación común.
    # Todo lo demás se filtrará (o se reemplazará por un espacio para evitar unir palabras).
    cleaned_chars = []
    allowed_punctuation = ".,;:!?¡¿-'()[]{}<>/@#%&=+*_~\\$" # Ajustar según se necesite
    
    for char in text:
        category = unicodedata.category(char)[0] # Obtiene la categoría principal (L, N, P, Z, S, C)
        if category in ['L', 'N']: # Letras (L), Números (N)
            cleaned_chars.append(char)
        elif category == 'Z': # Separadores (Zs = espacio)
            cleaned_chars.append(' ') # Asegurar que solo sea espacio simple
        elif char in allowed_punctuation: # Puntuación permitida
            cleaned_chars.append(char)
        elif category == 'P': # Otra puntuación no en la lista blanca
            cleaned_chars.append(' ') # Reemplazar por espacio
            
        # Los caracteres de otras categorías (S: Símbolos no emojis, C: Control, M: Marcas)
        # se omiten o se reemplazan por espacio si se quiere ser más cauto.
        # else:
        #     cleaned_chars.append(' ') # Opcional: reemplazar no deseados por espacio
            
    text = "".join(cleaned_chars)
    
    # 6. Eliminar separadores largos (de guiones o underscores) que puedan haberse formado o existido
    text = re.sub(r'[-_]{3,}', ' ', text)
    
    # 7. Reemplazar múltiples espacios (posiblemente introducidos por los reemplazos) por uno solo
    text = re.sub(r'\s+', ' ', text)
    
    # 8. Devolver en minúsculas y sin espacios extra al inicio/final
    return text.strip().lower()
    
# ------------------------------------------------------
# Liberar memoria GPU (si es necesario, aunque con `del model` y `empty_cache` suele ser suficiente)
# ------------------------------------------------------
def clear_gpu_memory():
    '''Libera recursos GPU explícitamente.'''
    global model
    if 'model' in globals() or 'model' in locals():
        try:
            del model
            print("Variable 'model' eliminada.")
        except NameError:
            print("Variable 'model' no existía para eliminar.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        #garbage collector:
        import gc
        gc.collect()
        print("Memoria GPU y caché liberadas.")
    else:
        print("No hay GPU disponible para liberar memoria.")


# ------------------------------------------------------
# Generación de texto con fallback a CPU mejorado
# ------------------------------------------------------
def generate_text_robust(prompt: str,
                         max_new_tokens: int = 50, 
                         temperature: float = 0.9,
                         top_k: int = 50,
                         top_p: float = 0.95,
                         current_device: str = DEVICE, 
                         local_model = None
                         ) -> str:
    '''Genera texto; si falla en GPU, cae a CPU.'''
    if local_model is None:
        raise ValueError("El modelo no ha sido pasado a generate_text_robust")

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to(current_device)
    
    if inputs['input_ids'].shape[1] == 0:
        print("Advertencia: Input IDs vacíos después de tokenizar el prompt. Usando prompt de relleno.")
        inputs = tokenizer(" ", return_tensors='pt').to(current_device) # Usar un tokenizador de relleno

    # La longitud de la salida se controla directamente con max_new_tokens en model.generate
    # por lo que no es estrictamente necesario precalcular generate_max_length aquí
    # si transformers.__version__ es lo suficientemente reciente.

    try:
        with torch.no_grad():
            outputs = local_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens, 
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            )
        # Decodificar solo los tokens nuevos generados
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    except RuntimeError as e:
        if 'out of memory' in str(e).lower() and current_device == 'cuda':
            print(f"Error de OOM en {current_device}, reintentando en CPU: {e}")
            
            print("Moviendo modelo a CPU para este intento...")
            # Para evitar modificar el estado del modelo global de forma inesperada en un hilo,
            # idealmente se debería tener una instancia separada para CPU o no mover el modelo global.
            # Aquí, si local_model es el modelo global, se moverá.
            cpu_model_instance = local_model.to('cpu') 
            inputs_cpu = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512).to('cpu') # Enviar tensores de input a CPU
            
            with torch.no_grad():
                outputs_cpu = cpu_model_instance.generate(
                    **inputs_cpu,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(outputs_cpu[0][inputs_cpu['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Si el modelo movido a CPU era el modelo global y originalmente estaba en CUDA,
            # se podría considerar devolverlo a CUDA. Sin embargo, esto puede ser complejo
            # con multithreading. Es más seguro que el estado del modelo global se maneje fuera
            # de esta función o que se use una copia.
            if current_device == 'cuda' and local_model is model: # 'model' es el modelo global
                 print("ADVERTENCIA: El modelo global se movió a CPU debido a un OOM. Para futuras generaciones en GPU, puede necesitar ser movido de nuevo a CUDA explícitamente o recargado.")
                 # model.to('cuda') # Descomentar con precaución si se entiende el impacto en otros hilos.
            elif current_device == 'cuda' and cpu_model_instance is not local_model: # Si se hizo una copia (no es el caso aquí)
                local_model.to('cuda') # Devuelve la copia original a cuda (no aplica a la lógica actual)

        else:
            print(f"Error no OOM ({type(e).__name__}) durante la generación en {current_device}: {e}")
            return "ERROR_GENERACION_RUNTIME" # Devolver un placeholder
            
    return clean_text(generated_text)

# ------------------------------------------------------
# Generadores spam/ham con prompts mejorados: No son estructuras, son prompts semilla
# ------------------------------------------------------
SPAM_PROMPTS = [
    "ESCRIBE UN CORREO ELECTRÓNICO URGENTE DE SPAM CON UNA OFERTA INCREÍBLE:",
    "CREA UN MENSAJE DE PHISHING QUE PIDA AL USUARIO ACTUALIZAR SUS DATOS BANCARIOS:",
    "GENERA UN ANUNCIO ENGAÑOSO SOBRE UN PRODUCTO MILAGROSO:",
    "TEXTO DE SPAM: ¡Felicidades! Has ganado un premio. Reclámalo aquí:",
    "Asunto: ¡Oportunidad Única! Mensaje: Estimado cliente, tenemos una oferta exclusiva para usted",
    "URGENTE: Su cuenta requiere atención inmediata. Haga clic para verificar:",
    "Promoción exclusiva: Descuento del 90% solo por hoy. No se lo pierda:",
    "Ha sido seleccionado para recibir un regalo especial. Confirme sus datos para el envío:"
]

HAM_PROMPTS = [
    "ESCRIBE UN CORREO ELECTRÓNICO COTIDIANO ENTRE COLEGAS SOBRE EL PROYECTO ACTUAL:",
    "REDACTA UN MENSAJE INFORMAL DE UN AMIGO A OTRO PLANIFICANDO ALGO PARA EL FIN DE SEMANA:",
    "GENERA UN EMAIL PROFESIONAL PERO AMABLE PREGUNTANDO SOBRE EL ESTADO DE UNA TAREA:",
    "MENSAJE NORMAL Y CORRIENTE: Hola, ¿cómo estás? Quería preguntarte si",
    "Asunto: Reunión de equipo. Mensaje: Hola a todos, les recuerdo nuestra reunión de mañana a las 10am para discutir",
    "Confirmación de cita: Le recordamos su cita para el próximo martes a las 15:00.",
    "Consulta sobre el informe: ¿Podrías por favor revisar el borrador que te envié ayer?",
    "Saludos y actualización rápida: ¡Hola! Solo quería contarte que todo va bien por aquí."
]

# Nuevos parámetros para generate_text_robust
NEW_TOKENS_SPAM = 60 # Spam a veces es más largo
NEW_TOKENS_HAM = 40  # Ham puede ser más corto

def generate_spam_message() -> tuple:
    '''Devuelve (mensaje, 'spam').'''
    prompt = random.choice(SPAM_PROMPTS)
    # El modelo global `model` y `DEVICE` son accedidos implícitamente por `generate_text_robust`
    # si no se pasan explícitamente. Es mejor pasarlos.
    mensaje_spam = generate_text_robust(prompt, max_new_tokens=NEW_TOKENS_SPAM, local_model=model, current_device=DEVICE)
    print(f"SPAM Generado: {mensaje_spam[:100]}...") # Imprimir solo una parte para no llenar la consola
    return mensaje_spam, 'spam'

def generate_ham_message() -> tuple:
    '''Devuelve (mensaje, 'ham').'''
    prompt = random.choice(HAM_PROMPTS)
    mensaje_ham = generate_text_robust(prompt, max_new_tokens=NEW_TOKENS_HAM, local_model=model, current_device=DEVICE)
    print(f"HAM Generado: {mensaje_ham[:100]}...")
    return mensaje_ham, 'ham'

# ------------------------------------------------------
# Crear dataset con multithreading
# ------------------------------------------------------
def create_dataset(total: int,
                   spam_ratio: float,
                   max_workers: int = 4) -> pd.DataFrame:
    '''Genera un DataFrame con mensajes spam/ham.'''
    n_spam = int(total * spam_ratio)
    n_ham = total - n_spam
    
    # Asegurarse de que al menos hay un mensaje de cada tipo si el total es pequeño
    if total > 0 and spam_ratio > 0 and spam_ratio < 1:
        if n_spam == 0:
            n_spam = 1
            n_ham = max(0, total - 1)
        if n_ham == 0:
            n_ham = 1
            n_spam = max(0, total - 1)

    print(f"Generando {n_spam} mensajes de spam y {n_ham} mensajes de ham.")

    tasks = ([generate_spam_message] * n_spam) + ([generate_ham_message] * n_ham)
    random.shuffle(tasks)

    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func) for func in tasks]
        for i, fut in enumerate(as_completed(futures)):
            try:
                msg, label = fut.result()
                # Escapar comas internas ya no es tan necesario con csv.QUOTE_ALL,
                # pero no hace daño limpiar un poco más si se desea.
                # msg = msg.replace(',', ' ') # Considera si realmente necesitas esto
                results.append((msg, label))
                print(f"Progreso: {len(results)}/{total} mensajes generados.")
            except Exception as e:
                print(f"Error generando mensaje en el hilo: {e}")
                results.append(("ERROR_EN_GENERACION", "error")) # Añadir un placeholder
    return pd.DataFrame(results, columns=['mensaje', 'tipo'])

# ------------------------------------------------------
# Interacción con el usuario
# ------------------------------------------------------
def run_interactive():
    global model, tokenizer, DEVICE # Para poder referenciarlos y limpiarlos

    start_time_script = time.time()
    
    # print("Modelos en español disponibles: datificate/gpt2-small-spanish, DeepESP/gpt2-spanish, mrm8488/spanish-gpt2")
    while True:
        try:
            total_mensajes = int(input("Ingrese el número total de mensajes a generar (e.g., 100): "))
            if total_mensajes <= 0: raise ValueError
            break
        except ValueError:
            print("Por favor ingrese un entero positivo.")
    
    while True:
        try:
            ratio_spam = float(input("Ingrese la proporción de spam (0.0 a 1.0, e.g., 0.5 para 50%): "))
            if not 0.0 <= ratio_spam <= 1.0: raise ValueError
            break
        except ValueError:
            print("Por favor ingrese un valor entre 0.0 y 1.0.")
    
    # Hilos
    # Sugerir un número de hilos por defecto basado en CPU podría ser útil
    # import multiprocessing
    # default_workers = multiprocessing.cpu_count()
    default_workers = 4 # Un valor conservador
    while True:
        try:
            workers = int(input(f"Número de hilos para generación (e.g., {default_workers}): "))
            if workers <= 0: raise ValueError
            break
        except ValueError:
            print("Por favor ingrese un entero positivo.")

    print(f"\nGenerando {total_mensajes} mensajes con ~{ratio_spam*100:.1f}% spam usando {MODEL_NAME} en {DEVICE} con {workers} hilos...")
    
    generation_start_time = time.time()
    df = create_dataset(total_mensajes, ratio_spam, workers)
    generation_end_time = time.time()
    
    print(f"\n--- Vista previa del Dataset ({len(df)} filas) ---")
    # Usar display si está en un entorno como Jupyter/Colab, sino print
    try:
        display(df.head())
        if len(df) > 5:
            display(df.sample(min(5, len(df)-5))) # Muestra algunas aleatorias también
    except NameError: # 'display' no está definido
        print(df.head().to_string(index=False))
        if len(df) > 5:
            print("--- Muestra aleatoria ---")
            print(df.sample(min(5, len(df)-5)).to_string(index=False))


    filename = f'dataset_spam_ham_{MODEL_NAME.replace("/", "_")}_{total_mensajes}_{int(ratio_spam*100)}.csv'
    df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8')
    print(f"\nDataset guardado en '{filename}'. Se usaron comillas para proteger comas internas.")
    
    print(f"Tiempo de generación del dataset: {generation_end_time - generation_start_time:.2f} segundos.")
    
    # Limpieza final
    print("Limpiando recursos...")
    if 'model' in globals() or 'model' in locals(): # Verificar si la variable existe
        clear_gpu_memory() # Llama a la función de limpieza explícita
        # Borrar explícitamente las variables globales del modelo y tokenizador si ya no se necesitan
        del globals()['model'] 
        del globals()['tokenizer']
        print("Modelo y tokenizador eliminados de la memoria global.")

    end_time_script = time.time()
    print(f"Proceso completado. Tiempo total del script: {end_time_script - start_time_script:.2f} segundos.")

if __name__ == '__main__':
    try:
        run_interactive()
    except Exception as e:
        print(f"Se produjo un error fatal en el script: {e}")
        # Intentar limpiar GPU incluso si hay un error general
        if torch.cuda.is_available():
            print("Intentando liberar memoria GPU después del error...")
            clear_gpu_memory()
    finally:
        print("Script finalizado.")
