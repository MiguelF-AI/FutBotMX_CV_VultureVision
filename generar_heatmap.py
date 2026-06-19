import cv2
import json
import numpy as np

RUTA_JSON = "partido_data_suave.json" # Usamos los datos purificados
RUTA_SALIDA = "mapa_de_calor_tactico.png"

def generar_mapa_calor():
    print(" Cargando datos tácticos...")
    with open(RUTA_JSON, 'r') as f:
        datos = json.load(f)

    # Autocorrección de estructura
    if "timeline" in datos:
        timeline = datos["timeline"]
    elif "metadata" in datos:
        timeline = {k: v for k, v in datos.items() if k != "metadata"}
    else:
        timeline = datos

    # 1. CREAR LA CANCHA VIRTUAL
    # La cancha mide 1.58m x 2.19m. Vamos a escalarla x400 para tener una imagen HD
    escala = 400
    ancho_px = int(1.58 * escala)
    alto_px = int(2.19 * escala)

    # Matriz vacía para acumular el "calor" (flotante para que no tenga límite)
    capa_calor = np.zeros((alto_px, ancho_px), dtype=np.float32)

    print(" Calculando densidad térmica...")
    frames_procesados = 0

    for frame_str, datos_frame in timeline.items():
        robots = datos_frame.get("robots", []) if isinstance(datos_frame, dict) else datos_frame
        
        for robot in robots:
            metros = robot.get("pos_metros")
            if metros:
                x_m, y_m = metros[0], metros[1]
                
                # Convertir metros a píxeles en nuestro nuevo lienzo
                x_px = int(x_m * escala)
                y_px = int(y_m * escala)

                # Si el robot está dentro de la cancha, sumamos "calor" a esa zona
                if 0 <= x_px < ancho_px and 0 <= y_px < alto_px:
                    # Dibujamos un punto suave que se irá sumando si el robot se queda ahí
                    cv2.circle(capa_calor, (x_px, y_px), 25, 1, -1)
                    frames_procesados += 1

    print(f" Se procesaron {frames_procesados} posiciones de robots.")
    print(" Renderizando mapa visual...")

    # 2. PROCESAMIENTO VISUAL
    # Aplicamos un difuminado (Blur) muy agresivo para que los puntos se mezclen como nubes de calor
    capa_calor = cv2.GaussianBlur(capa_calor, (71, 71), 0)
    
    # Normalizamos los valores para que entren en el rango de colores de una imagen (0 a 255)
    capa_calor_norm = cv2.normalize(capa_calor, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Aplicamos el filtro de colores (Azul = Frío, Rojo = Caliente)
    heatmap_color = cv2.applyColorMap(capa_calor_norm, cv2.COLORMAP_JET)

    # 3. DIBUJAR LAS LÍNEAS DE LA CANCHA
    # Borde exterior blanco
    cv2.rectangle(heatmap_color, (0, 0), (ancho_px-1, alto_px-1), (255, 255, 255), 8)
    # Línea central
    cv2.line(heatmap_color, (0, int(alto_px/2)), (ancho_px, int(alto_px/2)), (255, 255, 255), 4)

    # Guardar la imagen
    cv2.imwrite(RUTA_SALIDA, heatmap_color)
    print(f" ¡ÉXITO! Mapa de calor generado. Abre el archivo: {RUTA_SALIDA}")

if __name__ == "__main__":
    generar_mapa_calor()