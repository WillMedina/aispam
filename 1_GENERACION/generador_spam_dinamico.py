import pandas as pd
import random
import numpy as np
import csv
import os

def cargar_datos_desde_csv(ruta_archivo):
    """
    Carga datos desde un archivo CSV y los convierte en diccionarios
    para ser utilizados en la generación de mensajes.
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(ruta_archivo):
            print(f"Archivo no encontrado: {ruta_archivo}")
            return {}
        
        # Cargar el archivo
        df = pd.read_csv(ruta_archivo, quoting=csv.QUOTE_ALL)
        
        # Convertir cada columna en una lista
        datos = {}
        for columna in df.columns:
            # Eliminar valores NaN o nulos
            datos[columna] = df[columna].dropna().tolist()
        
        return datos
    except Exception as e:
        print(f"Error al cargar el archivo {ruta_archivo}: {e}")
        return {}

def guardar_lista_como_csv(lista, nombre_columna, nombre_archivo):
    """
    Guarda una lista simple como un archivo CSV con una sola columna.
    """
    df = pd.DataFrame({nombre_columna: lista})
    df.to_csv(nombre_archivo, index=False, quoting=csv.QUOTE_ALL)

def crear_archivos_plantilla():
    """
    Crea archivos CSV de plantilla si no existen.
    Estos archivos contienen los datos básicos para generar mensajes.
    """
    # Datos para mensajes spam - Creamos cada columna por separado
    templates_spam = [
        "¡OFERTA EXCLUSIVA! Compra {producto} con {descuento}% de descuento. Solo por hoy. Responde YA o llama al {telefono}.",
        "Felicidades! Has sido seleccionado para recibir un {producto} GRATIS. Haz click en {url} para reclamar tu premio antes de {horas} horas.",
        "URGENTE: Tu {cuenta} ha sido {problema}. Verifica tus datos en {url} para evitar el bloqueo de tu cuenta.",
        "¡{casino} te regala {cantidad}€ para jugar sin depósito! Regístrate ahora en {url} con el código {codigo}.",
        "{banco}: Detectamos actividad sospechosa en tu cuenta. Verifica tu identidad en {url} o tu cuenta será suspendida."
    ]
    
    # Guardar las plantillas de spam
    guardar_lista_como_csv(templates_spam, "templates", "plantillas_templates_spam.csv")
    
    # Variables para mensajes spam - Guardamos cada una por separado
    guardar_lista_como_csv(["iPhone 15", "Smart TV 55\"", "PlayStation 5", "AirPods Pro", "MacBook Air"], 
                         "productos", "plantillas_productos_spam.csv")
    
    guardar_lista_como_csv(["50", "70", "80", "65", "90"], 
                         "descuentos", "plantillas_descuentos_spam.csv")
    
    guardar_lista_como_csv(["612345678", "698765432", "654321987", "623456789", "687654321"], 
                         "telefonos", "plantillas_telefonos_spam.csv")
    
    guardar_lista_como_csv(["bit.ly/premio-ya", "promocion-exclusiva.net", "ofertas-limitadas.com", "verificar-cuenta.net", "premios-online.es"], 
                         "urls", "plantillas_urls_spam.csv")
    
    guardar_lista_como_csv(["cuenta bancaria", "tarjeta de crédito", "cuenta de Netflix", "perfil de PayPal", "suscripción"], 
                         "cuentas", "plantillas_cuentas_spam.csv")
    
    guardar_lista_como_csv(["bloqueada", "suspendida", "comprometida", "vencida", "en riesgo"], 
                         "problemas", "plantillas_problemas_spam.csv")
    
    guardar_lista_como_csv(["BetCasino", "LuckyPlay", "GananciasYA", "FortunaOnline", "MegaApuestas"], 
                         "casinos", "plantillas_casinos_spam.csv")
    
    guardar_lista_como_csv(["100", "200", "50", "500", "1000"], 
                         "cantidades", "plantillas_cantidades_spam.csv")
    
    guardar_lista_como_csv(["BBVA", "Santander", "CaixaBank", "Sabadell", "Bankinter"], 
                         "bancos", "plantillas_bancos_spam.csv")
    
    guardar_lista_como_csv(["WIN2024", "PREMIO50", "VIP2024", "LUCKY777", "BONO100"], 
                         "codigos", "plantillas_codigos_spam.csv")
    
    guardar_lista_como_csv(["24", "12", "48", "6", "72"], 
                         "horas", "plantillas_horas_spam.csv")
    
    # Datos para mensajes normales (ham)
    templates_ham = [
        "Hola {nombre}, ¿podemos vernos a las {hora} en {lugar}? Tengo que comentarte algo sobre el {tema}.",
        "Te recuerdo que mañana tenemos {evento} a las {hora}. No olvides traer {cosa}.",
        "{nombre}, acabo de ver que {tienda} tiene ofertas en {producto}. ¿Te interesa que vayamos el {dia}?",
        "Muchas gracias por la {objeto} que me prestaste. Te la devuelvo el {dia} cuando nos veamos en {lugar}.",
        "Hola, soy {nombre}. ¿Has terminado el informe de {tema}? El jefe lo está pidiendo para mañana."
    ]
    
    # Guardar las plantillas de ham
    guardar_lista_como_csv(templates_ham, "templates", "plantillas_templates_ham.csv")
    
    # Variables para mensajes ham - Guardamos cada una por separado
    guardar_lista_como_csv(["Juan", "María", "Carlos", "Ana", "Pablo", "Elena", "Miguel", "Laura"], 
                         "nombres", "plantillas_nombres_ham.csv")
    
    guardar_lista_como_csv(["15:30", "18:00", "10:15", "20:00", "12:45"], 
                         "horas", "plantillas_horas_ham.csv")
    
    guardar_lista_como_csv(["la cafetería", "la oficina", "el parque", "el centro comercial", "mi casa"], 
                         "lugares", "plantillas_lugares_ham.csv")
    
    guardar_lista_como_csv(["proyecto", "presupuesto", "fiesta", "viaje", "reunión"], 
                         "temas", "plantillas_temas_ham.csv")
    
    guardar_lista_como_csv(["reunión", "clase de yoga", "cena", "partido de fútbol", "presentación"], 
                         "eventos", "plantillas_eventos_ham.csv")
    
    guardar_lista_como_csv(["los documentos", "tu portátil", "algo para picar", "el regalo", "los apuntes"], 
                         "cosas", "plantillas_cosas_ham.csv")
    
    guardar_lista_como_csv(["El Corte Inglés", "Media Markt", "Zara", "Carrefour", "Amazon"], 
                         "tiendas", "plantillas_tiendas_ham.csv")
    
    guardar_lista_como_csv(["ropa", "electrónica", "muebles", "libros", "alimentos"], 
                         "productos", "plantillas_productos_ham.csv")
    
    guardar_lista_como_csv(["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"], 
                         "dias", "plantillas_dias_ham.csv")
    
    guardar_lista_como_csv(["chaqueta", "libro", "cámara", "bolsa", "herramienta"], 
                         "objetos", "plantillas_objetos_ham.csv")
    
    # Datos para comportamientos de spam (cada tipo de behavior por separado)
    guardar_lista_como_csv(["ventas", "phishing", "estafa", "promoción", "sorteo", "urgencia"], 
                         "behavior1", "plantillas_behavior1.csv")
    
    guardar_lista_como_csv(["electrodomésticos", "tecnología", "banca", "apuestas", "servicios", "salud"], 
                         "behavior2", "plantillas_behavior2.csv")
    
    guardar_lista_como_csv(["descuento", "gratis", "limitado", "exclusivo", "verificación", "premio"], 
                         "behavior3", "plantillas_behavior3.csv")
    
    guardar_lista_como_csv(["llamada a la acción", "miedo", "engaño", "presión", "exageración"], 
                         "behavior4", "plantillas_behavior4.csv")
    
    guardar_lista_como_csv(["enlace sospechoso", "contacto directo", "información personal", "pago adelantado", "oferta irreal"], 
                         "behavior5", "plantillas_behavior5.csv")
    
    print("Archivos de plantillas creados exitosamente.")

def generar_mensaje_desde_plantilla(template, variables):
    """
    Genera un mensaje basado en una plantilla y un conjunto de variables.
    """
    # Diccionario para almacenar los valores a insertar
    valores = {}
    
    # Buscar todos los campos entre llaves en la plantilla
    import re
    campos = re.findall(r'\{([^}]+)\}', template)
    
    # Para cada campo, intentar encontrar un valor en los datos
    for campo in campos:
        if campo in variables and variables[campo]:
            valores[campo] = random.choice(variables[campo])
        else:
            valores[campo] = f"[{campo}]"  # Valor por defecto si no hay datos
    
    # Rellenar la plantilla
    try:
        mensaje = template.format(**valores)
    except KeyError as e:
        mensaje = f"Error al formatear mensaje: {str(e)}. Plantilla: {template}"
    
    return mensaje

def generar_mensaje_spam(datos, max_chars=400):
    """
    Genera un mensaje de spam basado en plantillas y datos cargados.
    """
    # Verificar que existan plantillas
    if 'templates' not in datos or not datos['templates']:
        return "Mensaje de spam de ejemplo (faltan plantillas)."
    
    # Seleccionar una plantilla aleatoria
    template = random.choice(datos['templates'])
    
    # Generar el mensaje
    mensaje = generar_mensaje_desde_plantilla(template, datos)
    
    # Asegurar que no exceda el máximo de caracteres
    if len(mensaje) > max_chars:
        mensaje = mensaje[:max_chars-3] + "..."
    
    return mensaje

def generar_mensaje_ham(datos, max_chars=400):
    """
    Genera un mensaje ham (normal) basado en plantillas y datos cargados.
    """
    # Verificar que existan plantillas
    if 'templates' not in datos or not datos['templates']:
        return "Mensaje normal de ejemplo (faltan plantillas)."
    
    # Seleccionar una plantilla aleatoria
    template = random.choice(datos['templates'])
    
    # Generar el mensaje
    mensaje = generar_mensaje_desde_plantilla(template, datos)
    
    # Asegurar que no exceda el máximo de caracteres
    if len(mensaje) > max_chars:
        mensaje = mensaje[:max_chars-3] + "..."
    
    return mensaje

def generar_dataset(num_registros, datos_spam, datos_ham, datos_behaviors, proporcion_spam=0.5, max_chars=400):
    """
    Genera un dataset completo de mensajes spam y ham.
    """
    datos = []
    
    num_spam = int(num_registros * proporcion_spam)
    num_ham = num_registros - num_spam
    
    # Generar mensajes de spam
    for _ in range(num_spam):
        mensaje = generar_mensaje_spam(datos_spam, max_chars)
        
        # Generar comportamientos relevantes
        num_behaviors = random.randint(2, 5)  # Entre 2 y 5 comportamientos
        behavior_values = []
        
        for i in range(5):
            behavior_key = f'behavior{i+1}'
            if i < num_behaviors and behavior_key in datos_behaviors and datos_behaviors[behavior_key]:
                behavior_values.append(random.choice(datos_behaviors[behavior_key]))
            else:
                behavior_values.append(None)
        
        datos.append([mensaje, "spam"] + behavior_values)
    
    # Generar mensajes ham
    for _ in range(num_ham):
        mensaje = generar_mensaje_ham(datos_ham, max_chars)
        datos.append([mensaje, "ham", None, None, None, None, None])
    
    # Crear DataFrame
    df = pd.DataFrame(datos, columns=["Mensaje", "Tipo", "behavior1", "behavior2", "behavior3", "behavior4", "behavior5"])
    
    # Mezclar el dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df

def cargar_todos_los_datos():
    """
    Carga todos los archivos CSV necesarios y los organiza en diccionarios.
    """
    # Verificar si existen los archivos necesarios
    archivos_necesarios = [
        'plantillas_templates_spam.csv',
        'plantillas_productos_spam.csv',
        'plantillas_descuentos_spam.csv',
        'plantillas_telefonos_spam.csv',
        'plantillas_urls_spam.csv',
        'plantillas_cuentas_spam.csv',
        'plantillas_problemas_spam.csv',
        'plantillas_casinos_spam.csv',
        'plantillas_cantidades_spam.csv',
        'plantillas_bancos_spam.csv',
        'plantillas_codigos_spam.csv',
        'plantillas_horas_spam.csv',
        'plantillas_templates_ham.csv',
        'plantillas_nombres_ham.csv',
        'plantillas_horas_ham.csv',
        'plantillas_lugares_ham.csv',
        'plantillas_temas_ham.csv',
        'plantillas_eventos_ham.csv',
        'plantillas_cosas_ham.csv',
        'plantillas_tiendas_ham.csv',
        'plantillas_productos_ham.csv',
        'plantillas_dias_ham.csv',
        'plantillas_objetos_ham.csv',
        'plantillas_behavior1.csv',
        'plantillas_behavior2.csv',
        'plantillas_behavior3.csv',
        'plantillas_behavior4.csv',
        'plantillas_behavior5.csv'
    ]
    
    # Verificar si todos los archivos existen
    archivos_faltantes = [archivo for archivo in archivos_necesarios if not os.path.exists(archivo)]
    if archivos_faltantes:
        print(f"Faltan los siguientes archivos: {', '.join(archivos_faltantes)}")
        print("Creando archivos de plantillas...")
        crear_archivos_plantilla()
    
    # Cargar datos de spam
    datos_spam = {}
    datos_spam['templates'] = cargar_datos_desde_csv('plantillas_templates_spam.csv').get('templates', [])
    datos_spam['producto'] = cargar_datos_desde_csv('plantillas_productos_spam.csv').get('productos', [])
    datos_spam['descuento'] = cargar_datos_desde_csv('plantillas_descuentos_spam.csv').get('descuentos', [])
    datos_spam['telefono'] = cargar_datos_desde_csv('plantillas_telefonos_spam.csv').get('telefonos', [])
    datos_spam['url'] = cargar_datos_desde_csv('plantillas_urls_spam.csv').get('urls', [])
    datos_spam['cuenta'] = cargar_datos_desde_csv('plantillas_cuentas_spam.csv').get('cuentas', [])
    datos_spam['problema'] = cargar_datos_desde_csv('plantillas_problemas_spam.csv').get('problemas', [])
    datos_spam['casino'] = cargar_datos_desde_csv('plantillas_casinos_spam.csv').get('casinos', [])
    datos_spam['cantidad'] = cargar_datos_desde_csv('plantillas_cantidades_spam.csv').get('cantidades', [])
    datos_spam['banco'] = cargar_datos_desde_csv('plantillas_bancos_spam.csv').get('bancos', [])
    datos_spam['codigo'] = cargar_datos_desde_csv('plantillas_codigos_spam.csv').get('codigos', [])
    datos_spam['horas'] = cargar_datos_desde_csv('plantillas_horas_spam.csv').get('horas', [])
    
    # Cargar datos de ham
    datos_ham = {}
    datos_ham['templates'] = cargar_datos_desde_csv('plantillas_templates_ham.csv').get('templates', [])
    datos_ham['nombre'] = cargar_datos_desde_csv('plantillas_nombres_ham.csv').get('nombres', [])
    datos_ham['hora'] = cargar_datos_desde_csv('plantillas_horas_ham.csv').get('horas', [])
    datos_ham['lugar'] = cargar_datos_desde_csv('plantillas_lugares_ham.csv').get('lugares', [])
    datos_ham['tema'] = cargar_datos_desde_csv('plantillas_temas_ham.csv').get('temas', [])
    datos_ham['evento'] = cargar_datos_desde_csv('plantillas_eventos_ham.csv').get('eventos', [])
    datos_ham['cosa'] = cargar_datos_desde_csv('plantillas_cosas_ham.csv').get('cosas', [])
    datos_ham['tienda'] = cargar_datos_desde_csv('plantillas_tiendas_ham.csv').get('tiendas', [])
    datos_ham['producto'] = cargar_datos_desde_csv('plantillas_productos_ham.csv').get('productos', [])
    datos_ham['dia'] = cargar_datos_desde_csv('plantillas_dias_ham.csv').get('dias', [])
    datos_ham['objeto'] = cargar_datos_desde_csv('plantillas_objetos_ham.csv').get('objetos', [])
    
    # Cargar datos de behaviors
    datos_behaviors = {}
    datos_behaviors['behavior1'] = cargar_datos_desde_csv('plantillas_behavior1.csv').get('behavior1', [])
    datos_behaviors['behavior2'] = cargar_datos_desde_csv('plantillas_behavior2.csv').get('behavior2', [])
    datos_behaviors['behavior3'] = cargar_datos_desde_csv('plantillas_behavior3.csv').get('behavior3', [])
    datos_behaviors['behavior4'] = cargar_datos_desde_csv('plantillas_behavior4.csv').get('behavior4', [])
    datos_behaviors['behavior5'] = cargar_datos_desde_csv('plantillas_behavior5.csv').get('behavior5', [])
    
    return datos_spam, datos_ham, datos_behaviors

def main():
    """Función principal que ejecuta el generador de dataset"""
    # Cargar todos los datos necesarios
    print("Cargando datos...")
    datos_spam, datos_ham, datos_behaviors = cargar_todos_los_datos()
    
    # Solicitar parámetros al usuario
    try:
        num_registros = int(input("Ingrese el número de registros a generar: "))
        proporcion_spam = float(input("Ingrese la proporción de mensajes spam (0.0 a 1.0): "))
        if proporcion_spam < 0 or proporcion_spam > 1:
            proporcion_spam = 0.5
            print("Proporción inválida. Usando valor por defecto (0.5)")
        
        max_chars = int(input("Ingrese la longitud máxima de los mensajes: "))
        if max_chars <= 0:
            max_chars = 400
            print("Longitud inválida. Usando valor por defecto (400)")
    except ValueError:
        print("Error en la entrada. Usando valores por defecto.")
        num_registros = 100
        proporcion_spam = 0.5
        max_chars = 400
    
    # Generar el dataset
    print(f"Generando dataset con {num_registros} registros ({proporcion_spam*100:.1f}% spam)...")
    dataset = generar_dataset(
        num_registros, 
        datos_spam,
        datos_ham,
        datos_behaviors,
        proporcion_spam, 
        max_chars
    )
    
    # Guardar el dataset
    nombre_archivo = 'dataset_spam_ham.csv'
    dataset.to_csv(nombre_archivo, index=False, quoting=csv.QUOTE_ALL)
    print(f"Dataset generado exitosamente: {nombre_archivo}")
    
    # Mostrar una muestra del dataset
    print("\nMuestra del dataset generado:")
    print(dataset.head(3))
    
    # Estadísticas del dataset
    print("\nEstadísticas del dataset:")
    print(f"Total de mensajes: {len(dataset)}")
    print(f"Mensajes spam: {len(dataset[dataset['Tipo'] == 'spam'])}")
    print(f"Mensajes ham: {len(dataset[dataset['Tipo'] == 'ham'])}")

if __name__ == "__main__":
    main()
