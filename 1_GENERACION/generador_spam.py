#!pip install pandas random numpy re
#codigo obsoleto
import pandas as pd
import random
import numpy as np
from datetime import datetime
import re

# Función para generar mensajes de spam
def generar_mensaje_spam(max_chars=400):
    # Plantillas de mensajes de spam
    templates = [
        "¡OFERTA EXCLUSIVA! Compra {producto} con {descuento}% de descuento. Solo por hoy. Responde YA o llama al {telefono}.",
        "Felicidades! Has sido seleccionado para recibir un {producto} GRATIS. Haz click en {url} para reclamar tu premio antes de {horas} horas.",
        "URGENTE: Tu {cuenta} ha sido {problema}. Verifica tus datos en {url} para evitar el bloqueo de tu cuenta.",
        "¡{casino} te regala {cantidad}€ para jugar sin depósito! Regístrate ahora en {url} con el código {codigo}.",
        "{banco}: Detectamos actividad sospechosa en tu cuenta. Verifica tu identidad en {url} o tu cuenta será suspendida."
    ]

    # Elementos para rellenar plantillas
    productos = ["iPhone 15", "Smart TV 55\"", "PlayStation 5", "AirPods Pro", "MacBook Air"]
    descuentos = ["50", "70", "80", "65", "90"]
    telefonos = ["612345678", "698765432", "654321987", "623456789", "687654321"]
    urls = ["bit.ly/premio-ya", "promocion-exclusiva.net", "ofertas-limitadas.com", "verificar-cuenta.net", "premios-online.es"]
    cuentas = ["cuenta bancaria", "tarjeta de crédito", "cuenta de Netflix", "perfil de PayPal", "suscripción"]
    problemas = ["bloqueada", "suspendida", "comprometida", "vencida", "en riesgo"]
    casinos = ["BetCasino", "LuckyPlay", "GananciasYA", "FortunaOnline", "MegaApuestas"]
    cantidades = ["100", "200", "50", "500", "1000"]
    bancos = ["BBVA", "Santander", "CaixaBank", "Sabadell", "Bankinter"]
    codigos = ["WIN2024", "PREMIO50", "VIP2024", "LUCKY777", "BONO100"]
    horas = ["24", "12", "48", "6", "72"]

    # Seleccionar una plantilla aleatoria
    template = random.choice(templates)

    # Rellenar la plantilla
    mensaje = template.format(
        producto=random.choice(productos),
        descuento=random.choice(descuentos),
        telefono=random.choice(telefonos),
        url=random.choice(urls),
        cuenta=random.choice(cuentas),
        problema=random.choice(problemas),
        casino=random.choice(casinos),
        cantidad=random.choice(cantidades),
        banco=random.choice(bancos),
        codigo=random.choice(codigos),
        horas=random.choice(horas)
    )

    # Asegurar que no exceda el máximo de caracteres
    if len(mensaje) > max_chars:
        mensaje = mensaje[:max_chars-3] + "..."

    return mensaje

# Función para generar mensajes ham (normales)
def generar_mensaje_ham(max_chars=400):
    templates = [
        "Hola {nombre}, ¿podemos vernos a las {hora} en {lugar}? Tengo que comentarte algo sobre el {tema}.",
        "Te recuerdo que mañana tenemos {evento} a las {hora}. No olvides traer {cosa}.",
        "{nombre}, acabo de ver que {tienda} tiene ofertas en {producto}. ¿Te interesa que vayamos el {dia}?",
        "Muchas gracias por la {objeto} que me prestaste. Te la devuelvo el {dia} cuando nos veamos en {lugar}.",
        "Hola, soy {nombre}. ¿Has terminado el informe de {tema}? El jefe lo está pidiendo para mañana."
    ]

    nombres = ["Juan", "María", "Carlos", "Ana", "Pablo", "Elena", "Miguel", "Laura"]
    horas = ["15:30", "18:00", "10:15", "20:00", "12:45"]
    lugares = ["la cafetería", "la oficina", "el parque", "el centro comercial", "mi casa"]
    temas = ["proyecto", "presupuesto", "fiesta", "viaje", "reunión"]
    eventos = ["reunión", "clase de yoga", "cena", "partido de fútbol", "presentación"]
    cosas = ["los documentos", "tu portátil", "algo para picar", "el regalo", "los apuntes"]
    tiendas = ["El Corte Inglés", "Media Markt", "Zara", "Carrefour", "Amazon"]
    productos = ["ropa", "electrónica", "muebles", "libros", "alimentos"]
    dias = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    objetos = ["chaqueta", "libro", "cámara", "bolsa", "herramienta"]

    template = random.choice(templates)

    mensaje = template.format(
        nombre=random.choice(nombres),
        hora=random.choice(horas),
        lugar=random.choice(lugares),
        tema=random.choice(temas),
        evento=random.choice(eventos),
        cosa=random.choice(cosas),
        tienda=random.choice(tiendas),
        producto=random.choice(productos),
        dia=random.choice(dias),
        objeto=random.choice(objetos)
    )

    if len(mensaje) > max_chars:
        mensaje = mensaje[:max_chars-3] + "..."

    return mensaje

# Diccionarios para comportamientos de spam
behaviors_spam = {
    "behavior1": ["ventas", "phishing", "estafa", "promoción", "sorteo", "urgencia"],
    "behavior2": ["electrodomésticos", "tecnología", "banca", "apuestas", "servicios", "salud"],
    "behavior3": ["descuento", "gratis", "limitado", "exclusivo", "verificación", "premio"],
    "behavior4": ["llamada a la acción", "miedo", "engaño", "presión", "exageración"],
    "behavior5": ["enlace sospechoso", "contacto directo", "información personal", "pago adelantado", "oferta irreal"]
}

# Función para generar el dataset
def generar_dataset(num_registros, proporcion_spam=0.5):
    datos = []

    num_spam = int(num_registros * proporcion_spam)
    num_ham = num_registros - num_spam

    # Generar mensajes de spam
    for _ in range(num_spam):
        mensaje = generar_mensaje_spam()

        # Generar comportamientos relevantes
        num_behaviors = random.randint(2, 5)  # Entre 2 y 5 comportamientos
        behavior_values = []

        for i in range(5):
            if i < num_behaviors:
                behavior_values.append(random.choice(behaviors_spam[f"behavior{i+1}"]))
            else:
                behavior_values.append(None)

        datos.append([mensaje, "spam"] + behavior_values)

    # Generar mensajes ham
    for _ in range(num_ham):
        mensaje = generar_mensaje_ham()
        datos.append([mensaje, "ham", None, None, None, None, None])

    # Crear DataFrame
    df = pd.DataFrame(datos, columns=["Mensaje", "Tipo", "behavior1", "behavior2", "behavior3", "behavior4", "behavior5"])

    # Mezclar el dataset
    df = df.sample(frac=1).reset_index(drop=True)

    return df

# Generar el dataset
dataset = generar_dataset(num_registros=1000000, proporcion_spam=0.6)

# Mostrar una muestra
print(dataset.head())

# Guardar el dataset
dataset.to_csv('dataset_spam_ham.csv', index=False)
print(f"Dataset generado con {len(dataset)} registros")
