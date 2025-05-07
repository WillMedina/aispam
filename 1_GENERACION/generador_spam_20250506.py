"""
Generador de Dataset Spam/Ham Optimizado v2.1
Incluye todas las mejoras sugeridas: normalización, behaviours contextuales, control de duplicados y validación avanzada
"""
import pandas as pd
import random
import re
import csv
import os
import json
from pathlib import Path
from unicodedata import normalize
from collections import defaultdict
from typing import Dict, List, Tuple

# Configuración global
CONFIG = {
    "encoding": "utf-8-sig",
    "max_caracteres": 400,
    "max_intentos_unicos": 100,
    "prob_errores_ortograficos": 0.3,
    "behaviors": {
        "phishing": ["cuenta", "verificar", "urgente", "bloqueada", "contraseña"],
        "promocion": ["descuento", "gratis", "oferta", "exclusivo", "limitado"],
        "estafa": ["ganador", "premio", "loteria", "herencia", "transferencia"],
        "suplantacion": ["banco", "paypal", "netflix", "facebook", "google"],
        "urgente": ["inmediato", "ahora", "rapido", "ultima hora", "alerta"]
    }
}

class GeneradorUnico:
    def __init__(self):
        self.historico = defaultdict(set)
        self.combinaciones_generadas = defaultdict(set)
    
    def generar_combinacion_unica(self, categoria: str, variables: Dict) -> Tuple:
        intentos = 0
        while intentos < CONFIG["max_intentos_unicos"]:
            combinacion = tuple(random.choice(v) for v in variables.values())
            hash_combinacion = hash(combinacion)
            
            if hash_combinacion not in self.combinaciones_generadas[categoria]:
                self.combinaciones_generadas[categoria].add(hash_combinacion)
                return combinacion
            intentos += 1
        return None

def normalizar_texto(texto: str) -> str:
    texto = normalize('NFKC', texto).casefold()
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'
    }
    return ''.join(reemplazos.get(c, c) for c in texto)

def cargar_datos_desde_csv(ruta_archivo: str) -> Dict:
    try:
        if not Path(ruta_archivo).exists():
            return {}
        
        df = pd.read_csv(ruta_archivo, quoting=csv.QUOTE_ALL, encoding=CONFIG["encoding"])
        datos = {}
        for columna in df.columns:
            datos[columna] = [normalizar_texto(str(x)) for x in df[columna].dropna().tolist()]
        return datos
    except Exception as e:
        print(f"Error cargando {ruta_archivo}: {str(e)}")
        return {}

def crear_archivos_plantilla():
    plantillas = {
        "spam": {
            "templates": [
                "¡OFERTA EXCLUSIVA! Compra {producto} con {descuento}% de descuento. Solo por hoy. Responde YA o llama al {telefono}.",
                "Felicidades! Has sido seleccionado para recibir un {producto} GRATIS. Haz click en {url} para reclamar tu premio antes de {horas} horas.",
                "URGENTE: Tu {cuenta} ha sido {problema}. Verifica tus datos en {url} para evitar el bloqueo.",
                "¡{casino} te regala {cantidad}€! Registrate ahora en {url} con el codigo {codigo}.",
                "{banco}: Actividad sospechosa detectada. Verifica tu identidad en {url} o sera suspendida."
            ],
            "variables": {
                "productos": ["iphone 15", "smart tv 55", "playstation 5", "airpods pro", "macbook air"],
                "descuentos": ["50", "70", "80", "65", "90"],
                "telefonos": ["912345678", "986421735", "994880102", "973330210", "901555444"],
                "urls": ["bit.ly/premio-ya", "promocion-exclusiva.net", "verificar-cuenta.net"],
                "cuentas": ["cuenta bancaria", "tarjeta de credito", "cuenta de netflix"],
                "problemas": ["bloqueada", "suspendida", "comprometida"],
                "casinos": ["Casino Excalibur", "Casino Star", "Palacio Royal"],
                "cantidades": ["100", "200", "500", "1000"],
                "bancos": ["BBVA", "Interbank", "Banco de la Nacion"],
                "codigos": ["WIN2024", "PREMIO50", "BONO100"],
                "horas": ["24", "12", "48"]
            }
        },
        "ham": {
            "templates": [
                "Hola {nombre}, ¿quedamos a las {hora} en {lugar}? Tenemos que hablar del {tema}.",
                "Recuerda que mañana hay {evento} a las {hora}. No olvides traer {cosa}.",
                "{nombre}, {tienda} tiene ofertas en {producto}. ¿Vamos el {dia}?",
                "Gracias por la {objeto}. Te la devuelvo el {dia} en {lugar}.",
                "Hola, soy {nombre}. ¿Terminaste el informe de {tema}? Es urgente."
            ],
            "variables": {
                "nombres": ["Juan", "Maria", "Carlos", "Ana", "Pablo"],
                "horas": ["15:30", "18:00", "10:15", "20:00"],
                "lugares": ["la cafeteria", "la oficina", "el parque"],
                "temas": ["proyecto", "presupuesto", "fiesta", "viaje"],
                "eventos": ["reunion", "cena", "presentacion"],
                "cosas": ["documentos", "portatil", "regalo"],
                "tiendas": ["El Corte Ingles", "Media Markt", "Zara"],
                "productos": ["ropa", "electronica", "libros"],
                "dias": ["lunes", "martes", "miercoles", "viernes"],
                "objetos": ["chaqueta", "libro", "camara"]
            }
        }
    }

    for tipo in plantillas:
        dir_plantillas = Path("plantillas")
        dir_plantillas.mkdir(exist_ok=True)
        
        # Guardar templates
        plantilla_path = dir_plantillas / f"templates_{tipo}.csv"
        pd.DataFrame(plantillas[tipo]["templates"], columns=["template"]).to_csv(
            plantilla_path, index=False, encoding=CONFIG["encoding"], quoting=csv.QUOTE_ALL)
        
        # Guardar variables
        for variable, valores in plantillas[tipo]["variables"].items():
            var_path = dir_plantillas / f"{variable}_{tipo}.csv"
            pd.DataFrame(valores, columns=[variable]).to_csv(
                var_path, index=False, encoding=CONFIG["encoding"], quoting=csv.QUOTE_ALL)

def aplicar_variaciones(mensaje: str) -> str:
    if random.random() < CONFIG["prob_errores_ortograficos"]:
        sustituciones = {
            'a': ['@', '4'],
            'e': ['3'],
            'i': ['1', '!'],
            'o': ['0'],
            's': ['$', '5']
        }
        mensaje = ''.join(random.choice(sustituciones.get(c, [c])) for c in mensaje)
    
    if random.random() < 0.2:
        puntuacion = ['!', '!!', '...', '!!!', '']
        mensaje = mensaje.rstrip('.') + random.choice(puntuacion)
    
    return mensaje

def detectar_behaviors(mensaje: str) -> List[str]:
    mensaje = normalizar_texto(mensaje)
    detected = set()
    
    for behavior, keywords in CONFIG["behaviors"].items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', mensaje) for kw in keywords):
            detected.add(behavior)
    
    if not detected:
        detected.update(random.sample(list(CONFIG["behaviors"].keys()), k=2))
    
    return list(detected)[:5]

def generar_mensaje(tipo: str, datos: Dict, generador: GeneradorUnico) -> str:
    plantillas = datos[tipo].get("templates", [])
    if not plantillas:
        return ""
    
    template = random.choice(plantillas)
    variables_necesarias = re.findall(r'\{(.*?)\}', template)
    
    variables_disponibles = {}
    for var in variables_necesarias:
        var_data = datos[tipo].get(var, [])
        if var_data:
            variables_disponibles[var] = var_data
    
    combinacion = generador.generar_combinacion_unica(tipo, variables_disponibles)
    if not combinacion:
        return ""
    
    try:
        mensaje = template.format(**dict(zip(variables_disponibles.keys(), combinacion)))
        mensaje = aplicar_variaciones(mensaje)
        mensaje = normalizar_texto(mensaje)[:CONFIG["max_caracteres"]]
        
        if len(mensaje) < 15 or mensaje.count(' ') < 3:
            return generar_mensaje(tipo, datos, generador)
        
        return mensaje
    except KeyError as e:
        print(f"Error en template: Falta variable {e}")
        return ""

def cargar_todos_datos() -> Tuple[Dict, Dict]:
    datos = {"spam": {}, "ham": {}}
    
    for tipo in datos:
        # Cargar templates
        templates_path = Path("plantillas") / f"templates_{tipo}.csv"
        datos[tipo]["templates"] = cargar_datos_desde_csv(templates_path).get("template", [])
        
        # Cargar variables
        variables_files = list(Path("plantillas").glob(f"*_{tipo}.csv"))
        for var_file in variables_files:
            var_name = var_file.stem.split('_')[0]
            datos[tipo][var_name] = cargar_datos_desde_csv(var_file).get(var_name, [])
    
    return datos["spam"], datos["ham"]

def generar_dataset(num_registros: int, proporcion_spam: float) -> pd.DataFrame:
    spam_data, ham_data = cargar_todos_datos()
    generador = GeneradorUnico()
    dataset = []
    
    num_spam = int(num_registros * proporcion_spam)
    num_ham = num_registros - num_spam
    
    # Generar spam
    for _ in range(num_spam):
        mensaje = generar_mensaje("spam", {"spam": spam_data}, generador)
        if mensaje:
            behaviors = detectar_behaviors(mensaje)
            dataset.append([mensaje, "spam"] + behaviors)
    
    # Generar ham
    for _ in range(num_ham):
        mensaje = generar_mensaje("ham", {"ham": ham_data}, generador)
        if mensaje:
            dataset.append([mensaje, "ham"] + [None]*5)
    
    # Mezclar y crear DataFrame
    df = pd.DataFrame(dataset, columns=["Mensaje", "Tipo"] + list(CONFIG["behaviors"].keys())[:5])
    return df.sample(frac=1).reset_index(drop=True)

def main():
    if not Path("plantillas").exists():
        print("Creando plantillas iniciales...")
        crear_archivos_plantilla()
    
    try:
        num_registros = int(input("Número total de registros: "))
        proporcion_spam = float(input("Proporción de spam (0.1-0.9): "))
        proporcion_spam = max(0.1, min(0.9, proporcion_spam))
    except:
        num_registros = 1000
        proporcion_spam = 0.5
    
    df = generar_dataset(num_registros, proporcion_spam)
    output_file = "dataset_spam_ham_optimizado.csv"
    df.to_csv(output_file, index=False, encoding=CONFIG["encoding"], quoting=csv.QUOTE_ALL)
    
    print(f"\nDataset generado: {output_file}")
    print("Resumen:")
    print(f"- Total mensajes: {len(df)}")
    print(f"- Spam: {len(df[df['Tipo'] == 'spam'])}")
    print(f"- Ham: {len(df[df['Tipo'] == 'ham'])}")
    print("\nEjemplos de behaviors detectados:")
    print(df[list(CONFIG["behaviors"].keys())[:3]].dropna().head(3))

if __name__ == "__main__":
    main()
