import json
import cv2
import numpy as np

# Rutas de archivos
ENTRADA_JSON = "tracking_final.json"
SALIDA_JSON = "partido_data.json"

# ==========================================
# 1. MATRIZ DE CALIBRACIÓN (HOMOGRAFÍA)
# ==========================================

puntos_pixeles = np.array([
    [421, 236],   
    [955, 241],  
    [770, 1720],  
    [259, 1597]    
], dtype=np.float32)

# Dimensiones reales de las líneas(en metros)
ANCHO_METROS = 2.19  # 219 cm
ALTO_METROS = 1.58   # 158 cm

# Coordenadas exactas en metros de los 4 círculos rojos 
# Tomando la esquina superior izquierda de la línea blanca como (0,0)
puntos_metros = np.array([
    [0.29, 0.45],             # Círculo rojo Superior Izquierdo
    [1.29, 0.45],            # Círculo rojo Superior Derecho
    [1.29, 1.74],            # Círculo rojo Inferior Derecho
    [0.29, 1.74]              # Círculo rojo Inferior Izquierdo
], dtype=np.float32)

# Calculamos la Matriz de Deformación H
H = cv2.getPerspectiveTransform(puntos_pixeles, puntos_metros)

# ==========================================
# 2. PROCESAMIENTO DEL JSON
# ==========================================
def traducir_a_metros():
    print(" Cargando telemetría en píxeles...")
    try:
        with open(ENTRADA_JSON, 'r') as f:
            datos_pixeles = json.load(f)
    except FileNotFoundError:
        print(f" Error: No se encontró el archivo '{ENTRADA_JSON}'.")
        return

    datos_finales = {
        "metadata": {
            "partido_id": "copafutbotmx_procesado",
            "dimensiones_cancha_metros": {"ancho": ANCHO_METROS, "largo": ALTO_METROS}
        },
        "timeline": {}
    }

    print(" Transformando coordenadas mediante Homografía Lineal...")
    
    for frame_idx, contenido in datos_pixeles.items():
        frame_data = {"robots": [], "balon": None}
        
        # Procesar Robots
        for robot in contenido.get("robots", []):
            caja = robot["bbox_pixel"]
            
            # El punto de contacto con el suelo (centro inferior de la caja)
            x_centro = (caja[0] + caja[2]) / 2.0
            y_base = caja[3] 
            
            punto_pixel = np.array([[[x_centro, y_base]]], dtype=np.float32)
            punto_metro = cv2.perspectiveTransform(punto_pixel, H)[0][0]
            
            frame_data["robots"].append({
                "id": robot["id"],
                "pos_metros": [round(float(punto_metro[0]), 2), round(float(punto_metro[1]), 2)],
                "confianza": robot["confianza"]
            })
            
        # Procesar Pelota
        for bola in contenido.get("ball", []):
            caja = bola["bbox_pixel"]
            # Saca el Centro-Inferior (Bottom-Center):
            x_pixel = (caja[0] + caja[2]) / 2
            y_pixel = caja[3]

            punto_pixel = np.array([[[x_pixel, y_pixel]]], dtype=np.float32)
            punto_metro = cv2.perspectiveTransform(punto_pixel, H)[0][0]
            
            frame_data["balon"] = [round(float(punto_metro[0]), 2), round(float(punto_metro[1]), 2)]

        datos_finales["timeline"][frame_idx] = frame_data

    print(" Guardando contrato de datos final...")
    with open(SALIDA_JSON, 'w') as f:
        json.dump(datos_finales, f, indent=2)
        
    print(f" ¡Transformación 2D exitosa! El archivo '{SALIDA_JSON}' está calibrado milimétricamente.")

if __name__ == "__main__":
    traducir_a_metros()