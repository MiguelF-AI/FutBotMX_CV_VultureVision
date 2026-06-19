import json
import math
import matplotlib.pyplot as plt

RUTA_JSON = "partido_data_suave.json"
RUTA_GRAFICA = "intensidad_partido.png"

def calcular_intensidad():
    print("📂 Leyendo telemetría...")
    with open(RUTA_JSON, 'r') as f:
        datos = json.load(f)

    if "timeline" in datos:
        timeline = datos["timeline"]
    elif "metadata" in datos:
        timeline = {k: v for k, v in datos.items() if k != "metadata"}
    else:
        timeline = datos

    frames_ordenados = sorted(timeline.keys(), key=lambda x: int(x))
    FPS = 30 # Asumimos que tu video es de 30 fotogramas por segundo
    
    distancia_por_segundo = []
    distancia_acumulada_este_segundo = 0.0
    
    # Memoria a corto plazo para medir distancias (incluso si cambian de ID)
    posiciones_anteriores = {}

    print("📏 Calculando ritmo del partido...")
    for i, frame_str in enumerate(frames_ordenados):
        datos_frame = timeline[frame_str]
        robots = datos_frame.get("robots", []) if isinstance(datos_frame, dict) else datos_frame
        
        for robot in robots:
            r_id = robot.get("id")
            metros = robot.get("pos_metros")
            
            if r_id is not None and metros is not None:
                if r_id in posiciones_anteriores:
                    x1, y1 = posiciones_anteriores[r_id]
                    x2, y2 = metros[0], metros[1]
                    distancia = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Filtramos saltos irreales (errores de homografía)
                    if distancia < 0.5: 
                        distancia_acumulada_este_segundo += distancia
                
                posiciones_anteriores[r_id] = (metros[0], metros[1])
        
        # Cada 30 frames (1 segundo), guardamos el corte y reiniciamos
        if (i + 1) % FPS == 0:
            distancia_por_segundo.append(distancia_acumulada_este_segundo)
            distancia_acumulada_este_segundo = 0.0

    print("📊 Renderizando gráfica de intensidad temporal...")
    
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(distancia_por_segundo)), distancia_por_segundo, color='#06b6d4', linewidth=2)
    plt.fill_between(range(len(distancia_por_segundo)), distancia_por_segundo, color='#06b6d4', alpha=0.3)
    
    plt.title('Ritmo de Juego: Movimiento Total en la Cancha por Segundo', fontsize=16, fontweight='bold')
    plt.xlabel('Tiempo del Partido (Segundos)', fontsize=12)
    plt.ylabel('Metros Recorridos (Todos los robots combinados)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(RUTA_GRAFICA, dpi=300)
    print(f"✅ ¡ÉXITO! Gráfica generada: {RUTA_GRAFICA}")

if __name__ == "__main__":
    calcular_intensidad()