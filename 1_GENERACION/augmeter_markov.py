# pip install markovify
# pip install scikit-learn
import pandas as pd
import markovify 
import re
import unicodedata
# import argparse # Ya no es necesario
import csv # Para csv.QUOTE_ALL
from sklearn.utils import shuffle # Para mezclar el dataset final
import os # Para sugerir nombres de archivo

# --- MODULO 1: Limpieza de Texto ---
def clean_text_for_final_output(text_input: str) -> str:
    """
    Limpia exhaustivamente el texto: normaliza, quita emojis, ajusta comillas,
    quita comas, filtra caracteres no deseados y normaliza espacios.
    Convierte a minúsculas.
    """
    if not isinstance(text_input, str):
        return ""
    text = text_input
    try:
        text = unicodedata.normalize('NFC', text)
    except TypeError:
        return ""

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
            "\U00002B50"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(' ', text)
    except re.error:
        print("Advertencia: No se pudo compilar la regex de emojis. Algunos podrían permanecer.")
        text = ''.join(char for char in text if unicodedata.category(char)[0] not in ['So'])

    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('"', "'")
    text = text.replace('“', "'").replace('”', "'")
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace(',', ' ')

    cleaned_chars = []
    allowed_punctuation = ".;:!?¡¿-'()[]{}<>/@#%&=+*_~\\$"
    for char in text:
        category = unicodedata.category(char)[0]
        if category in ['L', 'N']:
            cleaned_chars.append(char)
        elif category == 'Z':
            cleaned_chars.append(' ')
        elif char in allowed_punctuation:
            cleaned_chars.append(char)
        elif category == 'P':
            cleaned_chars.append(' ')
    text = "".join(cleaned_chars)
    
    text = re.sub(r'[-_]{3,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip().lower()

def preprocess_text_for_markov_training(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# --- MODULO 1.5: Obtención de Parámetros por Consola ---
def get_interactive_paths() -> tuple[str, str]:
    """Obtiene las rutas de los archivos CSV de entrada y salida interactivamente."""
    while True:
        input_csv_path = input("Ingrese la ruta al archivo CSV original (ej: dataset.csv): ").strip()
        if input_csv_path:
            # Validación básica de existencia para guiar al usuario
            if not os.path.exists(input_csv_path):
                print(f"Advertencia: El archivo '{input_csv_path}' no parece existir. Por favor, verifique la ruta.")
                # Continuar de todas formas, load_original_dataset manejará el error final
            break
        print("La ruta del archivo de entrada no puede estar vacía.")

    default_output_name = ""
    if '.' in input_csv_path:
        base, ext = input_csv_path.rsplit('.', 1)
        default_output_name = f"{base}_augmented_markov.{ext}"
    else:
        default_output_name = f"{input_csv_path}_augmented_markov.csv"

    while True:
        output_csv_path = input(f"Ingrese la ruta para el nuevo CSV aumentado (default: {default_output_name}): ").strip()
        if not output_csv_path:
            output_csv_path = default_output_name
        if output_csv_path: # Si hay default o el usuario ingresó algo
            break
        # Este print es por si se quita el default y el usuario no ingresa nada
        print("La ruta del archivo de salida no puede estar vacía.")
        
    print(f"\nUsando archivo de entrada: {input_csv_path}")
    print(f"Se guardará el resultado en: {output_csv_path}")
    return input_csv_path, output_csv_path

def get_interactive_markov_parameters() -> tuple[int, str]:
    """Obtiene los parámetros para el modelo Markovify interactivamente."""
    default_state_size = 2
    while True:
        try:
            state_size_str = input(f"Ingrese el tamaño del estado para los modelos Markov (entero positivo, default: {default_state_size}): ").strip()
            if not state_size_str:
                state_size = default_state_size
                break
            state_size = int(state_size_str)
            if state_size <= 0:
                raise ValueError("El tamaño del estado debe ser un entero positivo.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}")
    
    default_model_type = "NewlineText"
    allowed_model_types = ["NewlineText", "Text"]
    while True:
        model_type_str = input(f"Ingrese el tipo de modelo Markovify ({'/'.join(allowed_model_types)}, default: {default_model_type}): ").strip()
        if not model_type_str:
            model_type = default_model_type
            break
        
        # Comparación insensible a mayúsculas/minúsculas
        canonical_model_type = next((mtype for mtype in allowed_model_types if mtype.lower() == model_type_str.lower()), None)
        if canonical_model_type:
            model_type = canonical_model_type # Usar la forma canónica (con mayúsculas correctas)
            break
        print(f"Entrada inválida. Por favor, elija entre {', '.join(allowed_model_types)}.")
        
    print(f"Usando tamaño de estado Markov: {state_size}")
    print(f"Usando tipo de modelo Markovify: {model_type}\n")
    return state_size, model_type

# --- MODULO 2: Carga y Análisis del Dataset Original ---
def load_original_dataset(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
        print(f"Dataset original '{filepath}' cargado: {len(df)} filas.")
        if 'mensaje' not in df.columns or 'tipo' not in df.columns:
            print("Error: El CSV debe contener las columnas 'mensaje' y 'tipo'.")
            return None
        # Convertir la columna 'mensaje' a string para evitar problemas con tipos mixtos
        df['mensaje'] = df['mensaje'].astype(str)
        return df
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV original en '{filepath}'.")
        return None
    except Exception as e:
        print(f"Error al cargar el CSV original: {e}")
        return None

def calculate_original_proportions(df: pd.DataFrame) -> tuple[float, float]:
    if df.empty or 'tipo' not in df.columns:
        return 0.5, 0.5
    
    type_counts = df['tipo'].value_counts(normalize=True)
    spam_ratio = type_counts.get('spam', 0.0)
    # Asegurar que si solo hay un tipo, el otro es 0 y suman 1
    # o si no hay ninguno de los dos (ej. otros tipos), se reparte 0.5
    if 'spam' in type_counts and 'ham' not in type_counts:
        ham_ratio = 0.0
    elif 'ham' in type_counts and 'spam' not in type_counts:
        ham_ratio = type_counts.get('ham', 0.0) # spam_ratio ya es 0.0
    elif 'spam' in type_counts and 'ham' in type_counts:
        ham_ratio = type_counts.get('ham', 0.0)
    else: # Ni spam ni ham presentes, o df vacío de estos tipos
        return 0.5, 0.5

    # Re-evaluar por si solo había un tipo
    if spam_ratio + ham_ratio < 1.0 and (spam_ratio > 0 or ham_ratio > 0):
        if spam_ratio == 0: ham_ratio = 1.0
        elif ham_ratio == 0: spam_ratio = 1.0
        # Si ambos son >0 pero <1 (otros tipos presentes), normalizar entre ellos
        # Esto no debería pasar si solo hay spam/ham. Si hay otros tipos, esta lógica es simple.
        # Para este script, asumimos que 'tipo' solo tiene 'spam' y 'ham' o es manejado.
        
    print(f"Proporciones originales en el dataset cargado: SPAM={spam_ratio:.2f}, HAM={1-spam_ratio:.2f}") # Asumiendo que el resto es HAM
    return spam_ratio, (1-spam_ratio) # O ham_ratio si se calculó más robustamente

# --- MODULO 3: Configuración del Usuario para Aumento ---
def get_user_augmentation_config(default_spam_ratio: float) -> tuple[int, float]:
    while True:
        try:
            total_new_messages_str = input("Ingrese el número TOTAL de nuevos mensajes a generar (ej: 1000): ").strip()
            if not total_new_messages_str: # Si el usuario solo presiona Enter
                 print("Debe ingresar un número. Intente de nuevo.")
                 continue
            total_new_messages = int(total_new_messages_str)
            if total_new_messages <= 0:
                raise ValueError("El número debe ser positivo.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}. Por favor, ingrese un entero positivo.")

    while True:
        try:
            prompt = (f"Ingrese la proporción de SPAM para los NUEVOS mensajes (0.0 a 1.0, "
                      f"default sugerido: {default_spam_ratio:.2f}): ")
            spam_ratio_new_str = input(prompt).strip()
            if not spam_ratio_new_str:
                spam_ratio_new = default_spam_ratio
            else:
                spam_ratio_new = float(spam_ratio_new_str)
            
            if not 0.0 <= spam_ratio_new <= 1.0:
                raise ValueError("La proporción debe estar entre 0.0 y 1.0.")
            break
        except ValueError as e:
            print(f"Entrada inválida: {e}. Por favor, ingrese un número decimal o Enter para el default.")
    
    print(f"Se generarán {total_new_messages} nuevos mensajes con ~{spam_ratio_new*100:.1f}% de spam.")
    return total_new_messages, spam_ratio_new

# --- MODULO 4: Entrenamiento y Generación con Markovify ---
def train_markov_model(texts: list[str], state_size: int = 2, model_type: str = "NewlineText") -> markovify.Text | None:
    if not texts:
        print("Advertencia: No hay textos para entrenar el modelo de Markov.")
        return None
    
    corpus = "\n".join(filter(None, texts)) # Filtrar strings vacíos antes de unir
    if not corpus.strip():
        print("Advertencia: Corpus vacío después de unir textos. No se puede entrenar.")
        return None

    print(f"  Entrenando modelo Markov ({model_type}, state_size={state_size}) con {len(texts)} bloques de texto...")
    try:
        # well_formed=False es más permisivo con frases no "perfectas"
        if model_type == "NewlineText":
            model = markovify.NewlineText(corpus, state_size=state_size, well_formed=False)
        else: # model_type == "Text"
            model = markovify.Text(corpus, state_size=state_size, well_formed=False)
        print("  Modelo entrenado.")
        return model
    except Exception as e:
        print(f"  Error al entrenar el modelo de Markov: {e}")
        return None

def generate_augmented_messages(model: markovify.Text | None, num_messages: int, label: str, tries_per_message: int = 100) -> list[dict]:
    if not model:
        print(f"No hay modelo de Markov disponible para generar mensajes de tipo '{label}'.")
        return []

    generated_messages_list = []
    print(f"Generando {num_messages} mensajes de tipo '{label}'...")
    generated_count = 0
    for i in range(num_messages):
        # Intenta generar una "oración". Los parámetros de overlap ayudan a la originalidad.
        new_message_text = model.make_sentence(
            tries=tries_per_message,
            max_overlap_ratio=0.7, 
            max_overlap_total=15
        )
        if new_message_text:
            generated_messages_list.append({'mensaje': new_message_text, 'tipo': label})
            generated_count +=1
        
        # Imprimir progreso cada 10% o cada mensaje si son pocos
        if num_messages < 20 or (i + 1) % (num_messages // 10 if num_messages >=10 else 1) == 0:
             print(f"  Intento {i+1}/{num_messages}. Generados hasta ahora: {generated_count} mensajes de '{label}'.")
    
    if generated_count < num_messages:
        print(f"Advertencia: Solo se pudieron generar {generated_count} de los {num_messages} mensajes solicitados para '{label}'. "
              "El corpus de entrenamiento podría ser pequeño o los parámetros de generación muy restrictivos.")
    return generated_messages_list

# --- MODULO 5: Combinación y Guardado del Dataset Final ---
def combine_and_prepare_final_df(original_df: pd.DataFrame, augmented_list: list[dict]) -> pd.DataFrame:
    df_orig_processed = original_df.copy()
    df_orig_processed['augmented'] = 0
    
    # LIMPIEZA FINAL de mensajes originales (si es que no se hizo antes de forma idéntica)
    # Esto asegura que el texto original también tenga el mismo formato final que el aumentado.
    print("Aplicando limpieza final a mensajes originales para consistencia...")
    df_orig_processed['mensaje'] = df_orig_processed['mensaje'].apply(clean_text_for_final_output)
    # Filtrar originales que quedaron vacíos después de la limpieza exhaustiva
    df_orig_processed = df_orig_processed[df_orig_processed['mensaje'].str.strip() != '']


    df_augmented = pd.DataFrame(augmented_list)
    if not df_augmented.empty:
        print("Aplicando limpieza final a mensajes generados por Markov...")
        df_augmented['mensaje'] = df_augmented['mensaje'].apply(clean_text_for_final_output)
        df_augmented['augmented'] = 1
        df_augmented = df_augmented[df_augmented['mensaje'].str.strip() != '']
    
    final_df = pd.concat([df_orig_processed, df_augmented], ignore_index=True)
    
    print("Mezclando el dataset final...")
    final_df = shuffle(final_df, random_state=42) 
    final_df = final_df.reset_index(drop=True)
    
    # Asegurar el orden de columnas
    final_columns = ['mensaje', 'tipo']
    if 'augmented' in final_df.columns:
        final_columns.append('augmented')
    
    # Mantener solo las columnas deseadas y en el orden correcto
    final_df = final_df[final_columns]

    return final_df

def save_final_dataset(df: pd.DataFrame, filepath: str):
    try:
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"Dataset final ({len(df)} filas) guardado exitosamente en: {filepath}")
    except Exception as e:
        print(f"Error al guardar el dataset final: {e}")

# --- MODULO 6: Orquestador Principal ---
def main():
    print("--- Iniciando Proceso de Aumento de Dataset con Markovify ---")

    # 1. Obtener rutas y parámetros de forma interactiva
    input_csv_path, output_csv_path = get_interactive_paths()
    state_size, markov_model_type = get_interactive_markov_parameters()

    # 2. Cargar dataset original
    df_original = load_original_dataset(input_csv_path)
    if df_original is None or df_original.empty:
        print("Proceso abortado: no se pudo cargar o el dataset original está vacío.")
        return

    # 3. Preprocesar textos para entrenamiento de Markov
    print("Preprocesando textos originales para entrenamiento de Markov...")
    df_original['markov_train_text'] = df_original['mensaje'].apply(preprocess_text_for_markov_training)
    df_original_trainable = df_original[df_original['markov_train_text'].str.strip() != ''].copy() # Usar .copy() para evitar SettingWithCopyWarning

    if df_original_trainable.empty:
        print("Proceso abortado: no quedaron mensajes válidos después del preprocesamiento para Markov.")
        return

    # 4. Separar textos y calcular proporciones originales
    spam_texts_for_train = df_original_trainable[df_original_trainable['tipo'] == 'spam']['markov_train_text'].tolist()
    ham_texts_for_train = df_original_trainable[df_original_trainable['tipo'] == 'ham']['markov_train_text'].tolist()
    
    original_spam_ratio, _ = calculate_original_proportions(df_original_trainable) # Usar el df con texto entrenable

    # 5. Obtener configuración del usuario para el aumento
    total_new, spam_ratio_new = get_user_augmentation_config(original_spam_ratio)
    
    num_new_spam = int(total_new * spam_ratio_new)
    num_new_ham = total_new - num_new_spam

    # 6. Entrenar modelos Markov
    print("\n--- Entrenamiento de Modelos Markov ---")
    markov_spam_model = None
    if spam_texts_for_train:
        print("Entrenando modelo para SPAM...")
        markov_spam_model = train_markov_model(spam_texts_for_train, state_size, markov_model_type)
    else:
        print("No hay textos de SPAM para entrenar el modelo correspondiente.")

    markov_ham_model = None
    if ham_texts_for_train:
        print("Entrenando modelo para HAM...")
        markov_ham_model = train_markov_model(ham_texts_for_train, state_size, markov_model_type)
    else:
        print("No hay textos de HAM para entrenar el modelo correspondiente.")

    # 7. Generar mensajes aumentados
    print("\n--- Generación de Mensajes Aumentados ---")
    all_augmented_messages = []
    if num_new_spam > 0:
        augmented_spam_messages = generate_augmented_messages(markov_spam_model, num_new_spam, 'spam')
        all_augmented_messages.extend(augmented_spam_messages)
    
    if num_new_ham > 0:
        augmented_ham_messages = generate_augmented_messages(markov_ham_model, num_new_ham, 'ham')
        all_augmented_messages.extend(augmented_ham_messages)
        
    if not all_augmented_messages and total_new > 0:
        print("No se generaron nuevos mensajes. Verifique los logs anteriores.")

    # 8. Combinar, limpiar y guardar
    print("\n--- Preparando Dataset Final ---")
    # Pasar el df_original sin la columna 'markov_train_text' y solo con las columnas necesarias
    df_original_for_concat = df_original[['mensaje', 'tipo']].copy()
    final_df = combine_and_prepare_final_df(df_original_for_concat, all_augmented_messages)
    
    if not final_df.empty:
        save_final_dataset(final_df, output_csv_path)
    else:
        print("No se generó ningún dato para guardar en el dataset final.")
    
    print("\n--- Proceso de Aumento de Dataset Finalizado ---")

if __name__ == '__main__':
    main()