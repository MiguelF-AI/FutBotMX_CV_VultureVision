import json
import numpy as np

RUTA_ENTRADA_JSON = "partido_data.json"
RUTA_SALIDA_JSON = "partido_data_suave.json"

VENTANA_SUAVIZADO = 5 
# NUEVO: Si un robot desaparece por menos de 60 cuadros (2 segundos), rellenamos el hueco.
MAX_HUECO_INTERPOLACION = 60 

def interpolar_trayectoria(frames, posiciones, confianzas, max_gap):
    """Rellena los fotogramas perdidos usando interpolación lineal."""
    nuevos_frames = []
    nuevas_posiciones = []
    nuevas_confianzas = []

    for i in range(len(frames) - 1):
        f_actual = frames[i]
        f_siguiente = frames[i+1]
        pos_actual = posiciones[i]
        pos_siguiente = posiciones[i+1]
        conf_actual = confianzas[i]

        # Guardamos el frame actual
        nuevos_frames.append(f_actual)
        nuevas_posiciones.append(pos_actual)
        nuevas_confianzas.append(conf_actual)

        hueco = f_siguiente - f_actual

        # Si hay un salto de frames, pero es menor al límite, creamos el puente matemático
        if 1 < hueco <= max_gap:
            paso_x = (pos_siguiente[0] - pos_actual[0]) / hueco
            paso_y = (pos_siguiente[1] - pos_actual[1]) / hueco

            for j in range(1, hueco):
                nuevos_frames.append(f_actual + j)
                # Calculamos la posición intermedia
                nuevas_posiciones.append([round(pos_actual[0] + paso_x * j, 3), round(pos_actual[1] + paso_y * j, 3)])
                # Le damos una confianza baja simulada para saber que fue generado por nosotros
                nuevas_confianzas.append(0.1) 

    # Agregamos el último frame de la lista
    if len(frames) > 0:
        nuevos_frames.append(frames[-1])
        nuevas_posiciones.append(posiciones[-1])
        nuevas_confianzas.append(confianzas[-1])

    return nuevos_frames, nuevas_posiciones, nuevas_confianzas

def aplicar_media_movil(trayectoria, ventana):
    """Aplica un filtro de media móvil centrada."""
    if len(trayectoria) < ventana:
        return trayectoria 
        
    coordenadas = np.array(trayectoria)
    coordenadas_suavizadas = np.copy(coordenadas)
    mitad = ventana // 2
    
    for i in range(mitad, len(coordenadas) - list(range(ventana))[-1] // 2):
        sub_arreglo = coordenadas[i - mitad : i + mitad + 1]
        coordenadas_suavizadas[i] = np.mean(sub_arreglo, axis=0)
        
    return coordenadas_suavizadas.tolist()

def purificar_telemetria():
    print(" Cargando contrato de datos...")
    with open(RUTA_ENTRADA_JSON, 'r') as f:
        datos_partido = json.load(f)

    metadata = datos_partido["metadata"]
    timeline = datos_partido["timeline"]
    frames_ordenados = sorted([int(k) for k in timeline.keys()])
    
    print(" Reconstruyendo series temporales...")
    trayectoria_robots = {}
    trayectoria_balon = {"frames": [], "posiciones": []}

    for f_idx in frames_ordenados:
        frame_str = str(f_idx)
        for robot in timeline[frame_str].get("robots", []):
            r_id = robot["id"]
            if r_id not in trayectoria_robots:
                trayectoria_robots[r_id] = {"frames": [], "posiciones": [], "confianzas": []}
            
            trayectoria_robots[r_id]["frames"].append(f_idx)
            trayectoria_robots[r_id]["posiciones"].append(robot["pos_metros"])
            trayectoria_robots[r_id]["confianzas"].append(robot["confianza"])
            
        pos_balon = timeline[frame_str].get("balon")
        if pos_balon is not None:
            trayectoria_balon["frames"].append(f_idx)
            trayectoria_balon["posiciones"].append(pos_balon)

    print(" Aplicando Imputación de Datos (Interpolación Lineal)...")
    for r_id in trayectoria_robots:
        frames_int, pos_int, conf_int = interpolar_trayectoria(
            trayectoria_robots[r_id]["frames"], 
            trayectoria_robots[r_id]["posiciones"], 
            trayectoria_robots[r_id]["confianzas"],
            MAX_HUECO_INTERPOLACION
        )
        trayectoria_robots[r_id]["frames"] = frames_int
        trayectoria_robots[r_id]["posiciones"] = pos_int
        trayectoria_robots[r_id]["confianzas"] = conf_int

    if len(trayectoria_balon["frames"]) > 0:
        frames_b, pos_b, _ = interpolar_trayectoria(
            trayectoria_balon["frames"], 
            trayectoria_balon["posiciones"], 
            [1]*len(trayectoria_balon["frames"]),
            MAX_HUECO_INTERPOLACION
        )
        trayectoria_balon["frames"] = frames_b
        trayectoria_balon["posiciones"] = pos_b

    print(f" Filtrando micro-vibraciones...")
    for r_id in trayectoria_robots:
        trayectoria_robots[r_id]["posiciones_suavizadas"] = aplicar_media_movil(trayectoria_robots[r_id]["posiciones"], VENTANA_SUAVIZADO)

    if len(trayectoria_balon["posiciones"]) > 0:
        trayectoria_balon["posiciones_suavizadas"] = aplicar_media_movil(trayectoria_balon["posiciones"], VENTANA_SUAVIZADO)

    print(" Reensamblando estructura de línea de tiempo...")
    nuevo_timeline = {}
    
    # IMPORTANTE: Aseguramos que el timeline contenga todos los frames posibles, incluyendo los interpolados
    todos_los_frames = set(frames_ordenados)
    for r_id, datos in trayectoria_robots.items():
        todos_los_frames.update(datos["frames"])
    todos_los_frames.update(trayectoria_balon["frames"])
    
    for f_idx in sorted(list(todos_los_frames)):
        nuevo_timeline[str(f_idx)] = {"robots": [], "balon": None}

    for r_id, datos in trayectoria_robots.items():
        for i in range(len(datos["frames"])):
            f_str = str(datos["frames"][i])
            pos_suave = datos["posiciones_suavizadas"][i]
            nuevo_timeline[f_str]["robots"].append({
                "id": r_id,
                "pos_metros": [round(pos_suave[0], 2), round(pos_suave[1], 2)],
                "confianza": datos["confianzas"][i]
            })

    for i in range(len(trayectoria_balon["frames"])):
        f_str = str(trayectoria_balon["frames"][i])
        pos_suave_balon = trayectoria_balon["posiciones_suavizadas"][i]
        nuevo_timeline[f_str]["balon"] = [round(pos_suave_balon[0], 2), round(pos_suave_balon[1], 2)]

    json_final = {
        "metadata": metadata,
        "timeline": nuevo_timeline,
        "eventos": datos_partido.get("eventos", []) 
    }

    print(" Exportando base de datos purificada...")
    with open(RUTA_SALIDA_JSON, 'w') as f:
        json.dump(json_final, f, indent=2)
        
    print(f" ¡Datos purificados con éxito! Archivo: '{RUTA_SALIDA_JSON}'")

if __name__ == "__main__":
    purificar_telemetria()