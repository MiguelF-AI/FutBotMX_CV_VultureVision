import json
import numpy as np
import supervision as sv

# Configuración de rutas
RUTA_ENTRADA_JSON = "output/coordenadas_partido.json"
RUTA_SALIDA_JSON = "tracking_final.json"

def procesar_rastreo_local():
    print(" Cargando coordenadas crudas desde el JSON...")
    try:
        with open(RUTA_ENTRADA_JSON, 'r') as f:
            registro_datos = json.load(f)
    except FileNotFoundError:
        print(f" Error: No se encontró el archivo '{RUTA_ENTRADA_JSON}' en esta carpeta.")
        return

    # track_thresh: Confianza mínima para mantener un objeto activo
    # lost_track_buffer: Cuántos cuadros esperar a que reaparezca un robot si se tapa
    tracker = sv.ByteTrack(
        track_activation_threshold=0.35,
        lost_track_buffer=45
    )

    json_tracking_final = {}
    
    # Ordenamos los frames numéricamente para asegurar la secuencia temporal del filtro
    frames_ordenados = sorted([int(k) for k in registro_datos.keys()])
    print(f" Procesando trayectorias en {len(frames_ordenados)} fotogramas registrados...")

    for frame_idx in frames_ordenados:
        frame_str = str(frame_idx)
        datos_frame = registro_datos[frame_str]

        cajas = datos_frame["cajas"]
        confianzas = datos_frame["confianzas"]
        clases = datos_frame["clases"]

        if len(cajas) == 0:
            # Si el frame está vacío, registramos la ausencia de objetos
            json_tracking_final[frame_idx] = {"robots": [], "ball": []}
            continue

        # Convertimos los arreglos nativos a objetos de detección de Supervision
        detecciones = sv.Detections(
            xyxy=np.array(cajas, dtype=np.float32),
            confidence=np.array(confianzas, dtype=np.float32),
            class_id=np.array(clases, dtype=np.int32)
        )

        # FILTRO DE KALMAN: Actualización y predicción de posiciones
        detecciones_trackeadas = tracker.update_with_detections(detecciones)

        robots_frame = []
        ball_frame = []

        # Extraemos los resultados con sus IDs asignados por el filtro
        for i in range(len(detecciones_trackeadas)):
            caja_coords = detecciones_trackeadas.xyxy[i].tolist()
            conf = float(detecciones_trackeadas.confidence[i])
            class_id = int(detecciones_trackeadas.class_id[i])
            
            # Si ByteTrack no logra asignar un ID en ese microsegundo, le ponemos -1
            track_id = int(detecciones_trackeadas.tracker_id[i]) if detecciones_trackeadas.tracker_id is not None else -1

            objeto_data = {
                "id": track_id,
                "bbox_pixel": [int(coord) for coord in caja_coords],
                "confianza": round(conf, 2)
            }

            # Clasificación de la estructura según tu mapeo indexado (0 = player/robot, 1 = orange ball)
            if class_id == 0:
                robots_frame.append(objeto_data)
            elif class_id == 1:
                ball_frame.append(objeto_data)

        # Empaquetamos la telemetría limpia de este fotograma
        json_tracking_final[frame_idx] = {
            "robots": robots_frame,
            "ball": ball_frame
        }

    print(" Guardando base de datos de telemetría purificada...")
    with open(RUTA_SALIDA_JSON, 'w') as f:
        json.dump(json_tracking_final, f, indent=2)
        
    print(f" ¡Pipeline completado con éxito! Archivo listo en: {RUTA_SALIDA_JSON}")

if __name__ == "__main__":
    procesar_rastreo_local()