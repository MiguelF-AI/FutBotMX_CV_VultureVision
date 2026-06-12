# FutBotMX_CV_VultureVision
.
# Plan Maestro del Proyecto: Scout 3D
## Copa FutBotMX — Categoría Profesional

Este documento contiene la especificación de arquitectura, diseño de datos, estructura de archivos y lógica algorítmica para el desarrollo del proyecto **Scout 3D**. El sistema está diseñado bajo un esquema desacoplado (Monorepo) que separa el procesamiento pesado de visión por computadora de la visualización interactiva tridimensional en tiempo real, garantizando escalabilidad, reproducibilidad y optimización de recursos de hardware local.

---

## 1. Arquitectura General del Sistema

El pipeline de datos está estructurado en dos grandes bloques tecnológicos autónomos conectados por un contrato de datos estricto (`partido_data.json`):

+---------------------------------------------------------------------------------------+<br>
|                                1. MOTOR DE EXTRACCIÓN                                 |<br>
|  [Video Crudo .mp4] -> [SAM 3 Fine-Tuned (Nube)] -> [Homografía y Análisis (Python)]  |<br>
+---------------------------------------------------------------------------------------+<br>
                                           |<br>
                                           v<br>
                             [Archivo de Datos: .json]<br>
                                           |<br>
                                           v<br>
+---------------------------------------------------------------------------------------+<br>
|                                2. VISUALIZADOR WEB 3D                                 |<br>
|  [Next.js Core] -> [Control de Línea de Tiempo] -> [Renderizado 3D (WebGL / R3F)]     |<br>
+---------------------------------------------------------------------------------------+<br>


### Bloque 1: Motor de Extracción (Ecosistema Python)
* **Entorno:** Google Colab (Inferencia de IA y entrenamiento pesado) + Entorno Local (Post-procesamiento geométrico).
* **Componentes:** Segment Anything Model 3 (Meta AI), OpenCV, SciPy.
* **Objetivo:** Analizar los videos analógicos de los partidos, segmentar las máscaras del balón y de los robots, normalizar sus posiciones métricas mediante homografía lineal, calcular la distribución espacial (Voronoi) y estructurar las reglas de eventos para exportar el archivo final.

### Bloque 2: Plataforma Web Interactiva (Ecosistema Frontend)
* **Entorno:** Servidor de desarrollo Node.js local con despliegue de producción optimizado en Vercel.
* **Componentes:** Next.js (React Framework), React Three Fiber (Three.js adaptado a componentes reactivos).
* **Objetivo:** Ingerir de forma dinámica el archivo `.json`, proveer controles fluidos de reproducción (reproducir, pausar, rebobinar), renderizar los robots como primitivas 3D ligeras a una tasa estable de cuadros por segundo, y pintar capas analíticas conmutables (Trails y Voronoi).

---

## 2. Estructura del Repositorio (Monorepo)

futbotmx-scout3d/<br>
│<br>
├── ai_engine/                             # Bloque 1: Procesamiento de IA y Datos (Python)<br>
│   ├── notebooks/<br>
│   │   └── sam3_fine_tuning_inference.ipynb # Cuaderno de Colab para Fine-Tuning e Inferencia de máscaras<br>
│   ├── scripts/<br>
│   │   ├── homography_translator.py       # Conversión de coordenadas de píxeles a metros reales<br>
│   │   ├── spatial_analysis.py            # Script encargado de calcular Voronoi y heurísticas lógicas<br>
│   │   └── json_builder.py                # Ensamblador del archivo JSON de salida<br>
│   ├── data/<br>
│   │   ├── raw_video.mp4                  # Muestra de video original de la Copa FutBotMX<br>
│   │   └── partido_data.json              # Fichero de datos estructurado resultante<br>
│   └── requirements.txt                   # Librerías necesarias (opencv-python, scipy, torch)<br>
│<br>
├── web_dashboard/                         # Bloque 2: Interfaz e Interacción 3D (Next.js)<br>
│   ├── app/<br>
│   │   ├── layout.tsx                     # Estructura e hidratación global de la aplicación<br>
│   │   ├── page.tsx                       # Dashboard principal con controles de UI y feed de eventos<br>
│   │   └── globals.css                    # Estilos CSS tailwind globales<br>
│   ├── components/<br>
│   │   ├── ui/                            # Componentes de la interfaz de usuario (Timeline, Selector, Logs)<br>
│   │   └── canvas3d/                      # Módulo de renderizado de Three.js<br>
│   │       ├── Scene.tsx                  # Configuración de Luces, Cámara Orbital y niebla de la escena<br>
│   │       ├── Field.tsx                  # Modelo geométrico plano de la cancha con líneas reglamentarias<br>
│   │       ├── RobotPrimitive.tsx         # Representación geométrica optimizada del robot (Caja + Textura)<br>
│   │       ├── Ball.tsx                   # Esfera naranja de alta velocidad con interpolación lineal<br>
│   │       └── VoronoiOverlay.tsx         # Mallas planas translúcidas de los polígonos de dominio espacial<br>
│   ├── public/<br>
│   │   └── data/                          # Directorio para almacenar los partidos disponibles<br>
│   │       ├── partido_final_t1.json<br>
│   │       └── partido_final_t2.json<br>
│   └── package.json                       # Dependencias de npm (@react-three/fiber, @react-three/drei, three, next)<br>
│<br>
├── README.md                              # Guía de instalación, reproducción y justificación del proyecto<br>
└── LICENSE                                # Licencia de código abierto de tipo MIT<br>


---

## 3. Pipeline de Datos y Extracción con Visión por Computadora

### 3.1. Estrategia de Fine-Tuning de SAM 3 (Estrategia Híbrida)
Para mitigar el ruido visual producido por las carcasas abiertas de los robots, cables y sombras complejas en la cancha, se entrena de forma personalizada la red fundamental:
1.  **Anotación Semántica Pura (Roboflow):** Se extraen aproximadamente 150-200 frames representativos distribuidos en una relación 70% del partido objetivo y 30% de otros partidos (con diferentes modelos de robots) para asegurar generalización. Se etiquetan únicamente dos clases:
    * `robot`: Abarca la estructura total del chasis (placa circular o cúbica, circuitos visibles y ruedas). Se excluyen rigurosamente las sombras proyectadas en el césped.
    * `balon`: Contorno ceñido de la pelota. En frames con desenfoque por velocidad extrema (*motion blur*), se etiqueta exclusivamente el núcleo de mayor densidad cromática.
2.  **Entrenamiento Independiente (PyTorch en Google Colab):** Se congela el Image Encoder masivo de SAM 3. Se entrena con los datos exportados únicamente el *Mask Decoder* ligero. Esto permite realizar el ajuste en pocas épocas, optimizando el uso de cómputo en la nube y generando un archivo de pesos (`.pth`) personalizado de alta precisión.
3.  **Clasificación de Equipos Post-Inferencia:** SAM 3 devuelve la máscara de segmentación de la clase genérica `robot`. El script `homography_translator.py` analiza el color promedio de los píxeles internos de esa máscara mediante OpenCV (mapeo en el espacio de color HSV). Si la firma de color dominante corresponde al rango del cyan, se cataloga dinámicamente como `equipo_local`, y si corresponde a tonos grises/oscuros, como `equipo_visitante`.

### 3.2. Traducción Espacial mediante Homografía Lineal
Dada la deformación de perspectiva natural del ángulo de la cámara del estadio, los píxeles de la pantalla `(u, v)` no guardan una relación lineal con los metros reales de la cancha `(x, y)`.

1.  **Matriz de Calibración:** Se seleccionan 4 puntos coplanares de referencia perfectamente visibles en el video (por ejemplo: la intersección de la línea media con las bandas laterales y las dos esquinas del área chica de una portería visible). Sus dimensiones en metros se conocen por el reglamento de la RoboCup.
2.  **Resolución de Esquinas Ocultas o Recortadas:** Al calcular la matriz de transformación proyectiva H de 3x3 con OpenCV usando los puntos internos conocidos, la proyección matemática se valida para todo el espacio del plano extendido. Si un robot camina por una esquina recortada de la transmisión de video pero su cuerpo sigue apareciendo en los píxeles capturados, el sistema calculará su coordenada métrica real exacta de forma limpia.
3.  **Filtrado Temporal:** Las coordenadas extraídas se someten a un filtro de media móvil en Python para eliminar micro-vibraciones causadas por la imprecisión del tracking cuadro a cuadro.

---

## 4. Algoritmos para Reglas Oficiales y Control de Ruido (Heurísticas)

El script `spatial_analysis.py` analiza secuencialmente la telemetría métrica en Python antes de construir el JSON para detectar situaciones reales de juego.

### 4.1. Falta de Progreso (Lack of Progress)
* **Condición Física:** El balón se queda atorado entre los chasis de los robots o contra los muros del campo, deteniendo la fluidez del partido.
* **Implementación Algorítmica:** Se evalúa la posición del balón en una ventana de tiempo móvil de N frames (por ejemplo, N = 90 frames para un video a 30 fps, equivalente a 3 segundos de juego).
* **Fórmula:** Se calcula la distancia euclidiana entre el inicio y el final de la ventana de análisis.
* **Acción:** Si la distancia d < 0.1 metros, el script infiere que el balón está bloqueado y genera un objeto en el arreglo de eventos globales de tipo `falta_progreso`.

### 4.2. Reposicionamiento Manual del Balón (Intervención del Árbitro)
* **Condición Física:** El árbitro levanta el balón detenido con la mano y lo coloca en un punto neutral central o de saque. En el video, esto se registra como una velocidad instantánea infinita (teletransportación).
* **Implementación Algorítmica:** Se calcula la velocidad escalar del balón entre el frame actual y el frame inmediatamente anterior.
* **Acción:** Si la velocidad calculada excede el límite físico del empuje de los robots comerciales (v > 5 m/s), el algoritmo descarta ese cuadro como movimiento natural, purga el vector de velocidad para que el rastro (*trail*) en el visualizador web no dibuje una línea cruzando la cancha, e inyecta un evento de tipo `intervencion_arbitro`.

### 4.3. Penalización y Salida de Robots de la Cancha
* **Condición Física:** Un robot es retirado físicamente de la mesa por los operadores debido a fallas en el hardware o por acumulación de faltas técnicas.
* **Implementación Algorítmica:** Si el tracker de SAM 3 deja de detectar un ID de robot específico durante más de M frames continuos (M = 90 cuadros / 3 segundos):
    * No se asume una falla de la Inteligencia Artificial. El script corta el flujo de coordenadas de ese robot en el timeline del JSON para los cuadros subsiguientes.
    * Crea una alerta estructurada en el listado de eventos: `"Robot X penalizado o fuera de la cancha"`.
    * **Comportamiento Web:** Cuando la aplicación de Next.js procesa los datos secuenciales en el canvas 3D y detecta que el objeto del robot específico no viene definido en el cuadro actual, muta automáticamente la propiedad del componente a `visible={false}`, ocultándolo de la escena de forma limpia hasta que vuelva a reaparecer en las lecturas de los píxeles del video.

---

## 5. Especificación del Contrato de Datos: `partido_data.json`

Para garantizar búsquedas directas en tiempo de ejecución de la escena 3D a una complejidad algorítmica óptima, la línea de tiempo se formatea como un objeto indexado por claves de string correspondientes al número del frame, evitando iteraciones pesadas de arreglos dentro del bucle de renderizado web.

```json
{
  "metadata": {
    "partido_id": "copafutbotmx_2026_final_medio1",
    "descripcion": "Análisis Scout 3D - Primer Tiempo Completo",
    "fps": 30,
    "total_frames": 21600,
    "dimensiones_cancha_metros": {
      "ancho": 6.0,
      "largo": 9.0
    },
    "equipos": {
      "local": {
        "nombre": "Bishops Knights C",
        "color_id": "cyan",
        "cantidad_robots": 3
      },
      "visitante": {
        "nombre": "i-bots",
        "color_id": "gris",
        "cantidad_robots": 3
      }
    }
  },
  "timeline": {
    "0": {
      "balon": [0.0, 0.0],
      "robots": {
        "local": [
          {"id": 1, "pos": [-1.5, 0.5]},
          {"id": 2, "pos": [-2.0, -1.0]},
          {"id": 5, "pos": [-0.5, 0.0]}
        ],
        "visitante": [
          {"id": 37, "pos": [1.5, -0.5]},
          {"id": 5, "pos": [2.0, 1.0]},
          {"id": 12, "pos": [0.5, 0.1]}
        ]
      },
      "voronoi": [
        {
          "equipo": "local",
          "vertices": [[0.0, 2.0], [-3.0, 2.0], [-3.0, -2.0], [0.0, -2.0]]
        },
        {
          "equipo": "visitante",
          "vertices": [[0.0, 2.0], [3.0, 2.0], [3.0, -2.0], [0.0, -2.0]]
        }
      ]
    }
  },
  "eventos": [
    {
      "frame": 0,
      "timestamp": "00:00",
      "tipo": "inicio",
      "descripcion": "Silbatazo inicial. Posiciones iniciales calibradas en 3D."
    },
    {
      "frame": 1470,
      "timestamp": "00:49",
      "tipo": "falta_progreso",
      "descripcion": "Falta de progreso detectada en media cancha. Balón estático."
    }
  ]
}
```

## 6. Implementación Frontend y Renderizado Gráfico 3D
La visualización interactiva utiliza aceleración gráfica nativa por WebGL en el navegador del usuario final, aislando el rendimiento de las limitantes de CPU.

### 6.1. Diseño de Componentes en React Three Fiber
* **Cancha (Field.tsx):** Un plano simple con dimensiones proporcionales a los metadatos. Las líneas reglamentarias se cargan como un mapa de textura único de baja resolución mapeado sobre la geometría para evitar el costo computacional de renderizar líneas vectoriales pesadas en vivo.

* **Robots (RobotPrimitive.tsx):** Se descarta la carga de modelos de mallas complejas .gltf de miles de polígonos. Los robots se construyen mediante primitivas geométricas básicas: una caja cúbica (<boxGeometry />) centrada, con mapas de texturas en sus seis caras que muestran el identificador numérico reglamentario del robot. Su frente integra un cilindro horizontal diminuto que denota el módulo pateador, emulando con fidelidad la estética del partido real.

* **Pelota (Ball.tsx):** Una primitiva de esfera (<sphereGeometry />) pintada de color naranja mate de alta visibilidad. Su posición se actualiza con interpolación lineal simple entre frames para asegurar suavidad visual en pantallas de alta tasa de refresco (144Hz+).

### 6.2. Optimización y Renderizado de Capas de Datos Avanzadas
* **Capa de Dominio Espacial (Voronoi):** El script de Python precalcula los polígonos basados en las posiciones de los robots utilizando scipy.spatial.Voronoi. El componente VoronoiOverlay.tsx mapea estas coordenadas de vértices directas en primitivas planas con un material translúcido (opacity={0.3}, transparent={true}) y colores indexados por equipo. Al estar precalculados en el JSON, el navegador web solo redibuja los nodos existentes, logrando tasas sostenidas de 60 FPS sin sobrecargar la memoria del sistema.

* **Rastros de Movimiento (Trails):** Se procesa en memoria local dentro de la web. Un hook personalizado mantiene un arreglo circular de los últimos 45 cuadros recorridos por cada robot. Se utiliza un componente <line /> o una geometría de tubo básica de Three.js conectando dichos puntos flotantes con una degradación gradual de opacidad, simulando un rastro dinámico de la estela táctica del jugador.

## 7. Estrategia de Entrega para el Comité Evaluador
Para cumplir estrictamente con las causales de validación y destacar con el máximo puntaje en la rúbrica de la categoría Profesional, se establece el siguiente protocolo de despliegue:

1. **Despliegue de Producción Automatizado:** La carpeta web_dashboard se vincula directamente a un pipeline de despliegue continuo en Vercel. Esto genera una URL de acceso universal pública. El jurado evaluador podrá evaluar de forma inmediata la escena 3D interactiva, rotar la cámara orbital y reproducir los partidos de prueba de manera nativa sin clonar el código localmente.

2. **Muestra Escalable de Datos:** Para optimizar los tiempos de desarrollo y respetar las capacidades de almacenamiento, se procesan y empaquetan los dos medios tiempos del partido clave detectado (24 minutos de datos estructurados de alta fidelidad). Se añade la documentación detallada en el README.md indicando los pasos exactos para que cualquier usuario pueda procesar un video nuevo e inyectarlo en la carpeta de datos públicos para que el selector web lo renderice de forma transparente.

3. **Generación de Entregables Obligatorios:**

  * **Video de Demostración (Máx. 2 minutos):** Captura de pantalla fluida mostrando la pantalla dividida (side-by-side) entre el video original segmentado por SAM 3 y la plataforma web interactiva "Scout 3D" en funcionamiento, explicando las heurísticas lógicas implementadas en Python para mitigar las fallas de oclusión visual y las reglas de RoboCup.
  
  * **Reel de Redes Sociales (Mín. 30 segundos):** Video dinámico con música y cortes rápidos enfocado puramente en la estética visual de la plataforma: la cámara orbitando los robots cúbicos en 3D, el encendido/apagado dinámico de los mapas de calor y la fluidez de los diagramas de Voronoi cubriendo la cancha en tiempo real. Enlace público embebido en el encabezado del archivo de documentación.
