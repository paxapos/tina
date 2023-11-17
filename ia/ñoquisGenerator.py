from PIL import Image, ImageDraw
import random
import os

# Diccionario de colores con nombres y valores RGB
colores = {
    '0': (255, 0, 0),
    '1': (0, 255, 0),
    '2': (0, 0, 255),
    '3': (255, 255, 0),
    '4': ( 255, 87, 51 ),
    '5': ( 149, 165, 166 ),
    '6': ( 99, 57, 116 ),
    '7': ( 212, 230, 241 ),
    '8': (255, 0, 255),
    '9': (0, 255, 255),
}

# Tamaño de la imagen
ancho = 224
altura = 224

# Obtener la ruta absoluta del directorio actual
directorio_actual = os.path.dirname(os.path.abspath(__file__))

# Crear la carpeta "ñoquis" si no existe
carpeta_nhoquis = os.path.join(directorio_actual, "ñoquis")
if not os.path.exists(carpeta_nhoquis):
    os.makedirs(carpeta_nhoquis)

# Función para generar una imagen con círculos y guardarla en la carpeta del color
def generar_imagen(color, cantidad):
    # Crear una carpeta con el nombre del color si no existe
    carpeta_color = os.path.join(carpeta_nhoquis, color)
    if not os.path.exists(carpeta_color):
        os.makedirs(carpeta_color)
    
    for i in range(cantidad):
        # Crear una nueva imagen en blanco
        imagen = Image.new("RGB", (ancho, altura), "white")
        dibujo = ImageDraw.Draw(imagen)
        
        # Obtener el valor RGB del color a partir del diccionario
        rgb = colores[color]
        
        # Generar círculos con el mismo color
        for _ in range(10):  # Puedes cambiar el número de círculos
            x1 = random.randint(0, ancho)
            y1 = random.randint(0, altura)
            radio = random.randint(10, 40)  # Puedes ajustar el rango de tamaños
            dibujo.ellipse([x1, y1, x1 + radio, y1 + radio], fill=rgb)
        
        # Guardar la imagen generada en la carpeta del color
        imagen.save(os.path.join(carpeta_color, f"{color}_imagen_{i+1}.png"))

# Parámetros
cantidad_imagenes = int(input("Ingrese la cantidad de imágenes a generar: "))

# Generar imágenes para cada color
for color in colores:
    generar_imagen(color, cantidad_imagenes)

print(f"Las imágenes se han guardado en el directorio: {carpeta_nhoquis}")
