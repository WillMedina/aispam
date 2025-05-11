"""
Generador de Dataset Spam/Ham Optimizado v3.0
Generación escalable con variables sintéticas y gestión inteligente de combinaciones
"""
import pandas as pd
import random
import re
import csv
from pathlib import Path
from unicodedata import normalize
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import itertools

# Configuración global
CONFIG = {
    "encoding": "utf-8-sig",
    "max_caracteres": 500,
    "max_intentos_unicos": 1000,
    "prob_errores_ortograficos": 0.3,
    "variables_sinteticas": 1000,
    "max_variables_por_template": 50,
    "behaviors": {
        "phishing": ["cuenta", "verificar", "urgente", "bloqueada", "contraseña"],
        "promocion": ["descuento", "gratis", "oferta", "exclusivo", "limitado"],
        "estafa": ["ganador", "premio", "loteria", "herencia", "transferencia"],
        "suplantacion": ["banco", "paypal", "netflix", "facebook", "google"],
        "urgente": ["inmediato", "ahora", "rapido", "ultima hora", "alerta"]
    },
    "rutas": {
        "plantillas": Path("plantillas"),
        "dataset": "dataset_spam_ham.csv"
    }
}

class GeneradorCombinaciones:
    def __init__(self):
        self.historico = defaultdict(lambda: defaultdict(set))
        self.generador_sintetico = itertools.count()
    
    def generar_combinacion(self, categoria: str, template: str, variables: Dict) -> Optional[tuple]:
        variables_necesarias = list(variables.keys())
        total_combinaciones = 1
        
        for var in variables_necesarias:
            total_combinaciones *= len(variables[var])
            if total_combinaciones == 0:
                return None

        for _ in range(min(CONFIG["max_intentos_unicos"], total_combinaciones)):
            combinacion = tuple(random.choice(variables[var]) for var in variables_necesarias)
            if self._combinacion_unica(categoria, template, combinacion):
                return combinacion
        
        # Fallback a datos sintéticos
        return self._generar_combinacion_sintetica(variables_necesarias)

    def _combinacion_unica(self, categoria: str, template: str, combinacion: tuple) -> bool:
        combo_hash = hash((template, combinacion))
        if combo_hash not in self.historico[categoria][template]:
            self.historico[categoria][template].add(combo_hash)
            return True
        return False

    def _generar_combinacion_sintetica(self, variables: List[str]) -> tuple:
        return tuple(f"{var}_{next(self.generador_sintetico)}" for var in variables)

def normalizar_texto(texto: str) -> str:
    texto = normalize('NFKC', texto).casefold()
    return texto.translate(str.maketrans({
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u'
    }))

def cargar_datos(ruta: Path) -> List[str]:
    try:
        if ruta.exists():
            df = pd.read_csv(ruta, quoting=csv.QUOTE_ALL, encoding=CONFIG["encoding"])
            return df.iloc[:, 0].dropna().apply(normalizar_texto).tolist()
        return []
    except Exception as e:
        print(f"Error cargando {ruta.name}: {str(e)}")
        return []

def crear_archivos_base():
    plantillas = {
        "spam": {
            "templates": [
                "¡OFERTA EXCLUSIVA! Compra {producto} con {descuento}% de descuento. Solo por hoy. Responde YA o llama al {telefono}.",
                "Felicidades! Has sido seleccionado para recibir un {producto} GRATIS. Haz click en {url} para reclamar tu premio antes de {hora} horas.",
                "URGENTE: Tu {cuenta} ha sido {problema}. Verifica tus datos en {url} para evitar el bloqueo.",
                "¡{casino} te regala {cantidad}€! Registrate ahora en {url} con el codigo {codigo}.",
                "{banco}: Actividad sospechosa detectada. Verifica tu identidad en {url} o sera suspendida."
            ],
            "variables": {
                "producto": ["iphone", "smart tv", "laptop", "tablet", "consola"],
                "descuento": ["50", "70", "80", "90"],
                "telefono": ["911111111", "922222222", "933333333"],
                "url": ["ofertas.com", "promociones.net", "ganadores.org"],
                "cuenta": ["bancaria", "de redes", "de streaming"],
                "problema": ["bloqueada", "suspendida", "hackeada"],
                "casino": ["Vegas", "Monaco", "Macao"],
                "cantidad": ["100", "200", "500"],
                "banco": ["BancoX", "BancoY", "BancoZ"],
                "codigo": ["GANAR", "PREMIO", "OFERTA"],
                "hora": ["24", "12", "48"]
            }
        },
        "ham": {
            "templates": [
                "Hola {nombre}, ¿quedamos a las {hora} en {lugar} para hablar del {tema}?",
                "Recuerda el {evento} de mañana a las {hora}. Necesitaremos {cosa}.",
                "{nombre}, hay ofertas en {tienda} de {producto}. ¿Vamos el {dia}?",
                "Gracias por el {objeto}. Te lo devuelvo el {dia} en {lugar}.",
                "¿Terminaste el informe de {tema}? Necesito revisarlo hoy."
            ],
            "variables": {
                "nombre": ["Ana", "Luis", "Marta", "Carlos"],
                "hora": ["10:00", "15:30", "19:00"],
                "lugar": ["cafetería", "oficina", "parque"],
                "tema": ["proyecto", "viaje", "fiesta"],
                "evento": ["reunión", "cena", "presentación"],
                "cosa": ["documentos", "portátil", "regalo"],
                "tienda": ["TiendaX", "TiendaY", "TiendaZ"],
                "producto": ["ropa", "libros", "electrónica"],
                "dia": ["lunes", "martes", "miércoles"],
                "objeto": ["libro", "llave", "cargador"]
            }
        }
    }

    for tipo, data in plantillas.items():
        dir_tipo = CONFIG["rutas"]["plantillas"]
        dir_tipo.mkdir(exist_ok=True)
        
        # Guardar templates
        pd.DataFrame({"template": data["templates"]}).to_csv(
            dir_tipo / f"templates_{tipo}.csv",
            index=False,
            encoding=CONFIG["encoding"],
            quoting=csv.QUOTE_ALL
        )
        
        # Guardar variables
        for var, valores in data["variables"].items():
            pd.DataFrame({var: valores}).to_csv(
                dir_tipo / f"{var}_{tipo}.csv",
                index=False,
                encoding=CONFIG["encoding"],
                quoting=csv.QUOTE_ALL
            )

def cargar_plantillas(tipo: str) -> Dict:
    datos = {"templates": [], "variables": defaultdict(list)}
    
    # Cargar templates
    templates_path = CONFIG["rutas"]["plantillas"] / f"templates_{tipo}.csv"
    datos["templates"] = cargar_datos(templates_path)
    
    # Cargar variables
    for var_file in CONFIG["rutas"]["plantillas"].glob(f"*_{tipo}.csv"):
        var_name = var_file.stem.split('_')[0]
        if var_name != "templates":
            datos["variables"][var_name] = cargar_datos(var_file)
    
    return datos

def generar_variables_sinteticas(tipo: str, variables_necesarias: List[str]) -> Dict[str, List[str]]:
    sinteticas = defaultdict(list)
    for var in variables_necesarias:
        base = var[:3].upper()
        sinteticas[var] = [f"{base}_SINT_{i}" for i in range(CONFIG["variables_sinteticas"])]
    return sinteticas

def aplicar_variaciones(texto: str) -> str:
    if random.random() < CONFIG["prob_errores_ortograficos"]:
        sustituciones = {'a':'@','e':'3','i':'1','o':'0','s':'$'}
        texto = ''.join(sustituciones.get(c, c) for c in texto)
    
    if random.random() < 0.3:
        texto += random.choice(['!','!!','...',''])
    
    return texto

def detectar_behaviors(texto: str) -> List[str]:
    texto = normalizar_texto(texto)
    detectados = set()
    for behavior, palabras in CONFIG["behaviors"].items():
        if any(palabra in texto for palabra in palabras):
            detectados.add(behavior)
    return list(detectados)[:3] or ["otros"]

def generar_mensaje(tipo: str, datos: Dict, generador: GeneradorCombinaciones) -> str:
    template = random.choice(datos["templates"])
    variables_necesarias = list(set(re.findall(r'\{(\w+)\}', template)))
    
    # Combinar datos reales y sintéticos
    variables = {}
    for var in variables_necesarias:
        reales = datos["variables"].get(var, [])
        sinteticas = [f"{var}_SINT_{i}" for i in range(len(reales), len(reales)+CONFIG["variables_sinteticas"])]
        variables[var] = reales + sinteticas[:CONFIG["max_variables_por_template"]]
    
    combinacion = generador.generar_combinacion(tipo, template, variables)
    if not combinacion:
        return ""
    
    try:
        mensaje = template.format(**dict(zip(variables_necesarias, combinacion)))
        mensaje = aplicar_variaciones(mensaje)
        return normalizar_texto(mensaje)[:CONFIG["max_caracteres"]]
    except KeyError as e:
        print(f"Error en template: Falta variable {e}")
        return ""

def detectar_behaviors(texto: str) -> List[str]:
    texto = normalizar_texto(texto)
    detectados = []
    for behavior, palabras in CONFIG["behaviors"].items():
        if any(palabra in texto for palabra in palabras):
            detectados.append(behavior)
    
    # Aseguramos 5 elementos, rellenando con None si es necesario
    return (detectados + [None]*(5))[:5]

def generar_dataset(num_registros: int, proporcion_spam: float) -> pd.DataFrame:
    datos_spam = cargar_plantillas("spam")
    datos_ham = cargar_plantillas("ham")
    generador = GeneradorCombinaciones()
    dataset = []
    
    num_spam = int(num_registros * proporcion_spam)
    num_ham = num_registros - num_spam
    
    # Generar spam
    for _ in range(num_spam):
        if (mensaje := generar_mensaje("spam", datos_spam, generador)):
            behaviors = detectar_behaviors(mensaje)
            dataset.append([mensaje, "spam"] + behaviors)
    
    # Generar ham
    for _ in range(num_ham):
        if (mensaje := generar_mensaje("ham", datos_ham, generador)):
            dataset.append([mensaje, "ham"] + [None]*5)
    
    # Cabeceras corregidas
    columnas = ["Mensaje", "Tipo"] + [f"B{i+1}" for i in range(5)]
    df = pd.DataFrame(dataset, columns=columnas)
    return df.sample(frac=1).reset_index(drop=True)

# ... (resto del código igual)

def main():
    if not CONFIG["rutas"]["plantillas"].exists():
        print("Creando plantillas base...")
        crear_archivos_base()
    
    try:
        num = int(input("Número de registros a generar: "))
        proporcion = float(input("Proporción de spam (0-1): "))
    except:
        num = 1000
        proporcion = 0.5
    
    df = generar_dataset(num, max(0.1, min(0.9, proporcion)))
    df.to_csv(CONFIG["rutas"]["dataset"], index=False, encoding=CONFIG["encoding"])
    
    print(f"\nDataset generado: {CONFIG['rutas']['dataset']}")
    print(df["Tipo"].value_counts())
    print("\nPrimeros registros:")
    print(df.head())

if __name__ == "__main__":
    main()
