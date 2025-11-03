# ---------------------------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------------------------
# cv2: OpenCV — manejo de cámara, imágenes y funciones de dibujo (líneas, círculos)
import cv2
# mediapipe: detección de pose humana 
import mediapipe as mp
# numpy: crear el canvas blanco donde se dibuja el stickman
import numpy as np
# deque: estructura de datos de tamaño limitado para almacenar historial de posiciones
from collections import deque

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DEL MODELO DE POSE (MediaPipe)
# ---------------------------------------------------------------------------
# mp_pose permite acceder a la solución 'pose' de MediaPipe.
# Esta solución detecta hasta 33 puntos clave (landmarks) en el cuerpo humano.
mp_pose = mp.solutions.pose

# Crear una instancia del detector/seguidor de pose.
# - min_detection_confidence: confianza mínima para aceptar una detección
# - min_tracking_confidence: confianza mínima para mantener el seguimiento
# Ajustar estos valores mejora robustez y/o sensibilidad.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE LA CÁMARA Y VIDEO DE SALIDA
# ---------------------------------------------------------------------------
# Abrimos la cámara por defecto (índice 0). Si no tienes cámara física, puedes
# sustituir 0 por la ruta a un archivo de vídeo para pruebas.
cap = cv2.VideoCapture(0)

# Obtener ancho y alto del frame para convertir coordenadas normalizadas a píxeles
frame_width = int(cap.get(3))   
frame_height = int(cap.get(4)) 

# Configurar VideoWriter para guardar la representación del stickman.
# Parámetros: archivo, codec (MJPG), fps, tamano (ancho, alto)
out = cv2.VideoWriter('stickman_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

# ---------------------------------------------------------------------------
# ESTRUCTURAS PARA EL HISTORIAL Y CONEXIONES
# ---------------------------------------------------------------------------
# Mantener un historial de las ultimas posiciones detectadas para dibujar el rastro.
# maxlen=30 limita el tamaño del historial, evitando uso excesivo de memoria.
positions_history = deque(maxlen=30)

# POSE_CONNECTIONS define qué puntos (landmarks) unir para formar el stickman.
# Cada tupla (a, b) indica que hay que dibujar una linea entre el landmark 'a' y 'b'.
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28)
]

# ---------------------------------------------------------------------------
# FUNCIÓN AUXILIAR: obtener landmarks en coordenadas píxel
# ---------------------------------------------------------------------------
# Esta función encapsula la lógica de detección y conversión:
# 1) Recibe un frame en BGR (formato de OpenCV)
# 2) Lo convierte a RGB (requisito de MediaPipe)
# 3) Ejecuta el proceso de MediaPipe para detectar pose
# 4) Si no hay detección devuelve None
# 5) Si hay detección, convierte cada landmark normalizado (0..1) en una tupla
#    (x, y) en píxeles multiplicando por el ancho y alto del frame

def get_landmarks(frame):
    # Convertir espacio de color BGR (OpenCV) a RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar con MediaPipe Pose
    results = pose.process(rgb)

    # Si no se detectan landmarks, devolvemos None para indicar ausencia de persona
    if not results.pose_landmarks:
        return None

    # Si hay landmarks, convertir cada landmark a coordenadas de píxeles
    h, w, _ = frame.shape
    return [(int(lm.x * w), int(lm.y * h)) for lm in results.pose_landmarks.landmark]

# ---------------------------------------------------------------------------
# BUCLE PRINCIPAL: captura, procesamiento, dibujo y guardado
# ---------------------------------------------------------------------------
try:
    # Ejecutar mientras la cámara esté abierta
    while cap.isOpened():
        # Capturar un frame de la cámara
        ret, frame = cap.read()
        # Si no se puede leer, salir del bucle (p. ej. cámara desconectada)
        if not ret:
            break

        # Obtener landmarks para el frame actual (lista de (x,y) o None)
        landmarks = get_landmarks(frame)

        # Preparar un canvas blanco del mismo tamaño que el frame para dibujar
        # el stickman y su historial. Esta imagen blanca se guardará en el vídeo.
        white_image = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Si hay landmarks detectados, dibujar el stickman y el historial
        if landmarks:
            # Añadir la posición actual al historial
            positions_history.append(landmarks)

            # Dibujar el historial (todas las posiciones guardadas) con un color
            # claro (amarillo). Esto crea el efecto de "rastro" detrás del movimiento.
            for prev in positions_history:
                for a, b in POSE_CONNECTIONS:
                    # prev[a] y prev[b] son tuplas (x, y) en píxeles
                    cv2.line(white_image, prev[a], prev[b], (0, 255, 255), 1)

            # Dibujar el stickman actual en rojo tanto en la imagen real como en
            # el canvas blanco que será guardado en el fichero.
            for a, b in POSE_CONNECTIONS:
                cv2.line(frame, landmarks[a], landmarks[b], (0, 0, 255), 2)
                cv2.line(white_image, landmarks[a], landmarks[b], (0, 0, 255), 2)

            # Marcar los puntos clave (articulaciones) como círculos rellenos
            # para hacer más evidente cada landmark.
            for x, y in landmarks:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                cv2.circle(white_image, (x, y), 4, (0, 0, 255), -1)

            # Guardar el canvas blanco (stickman con historial) en el fichero de salida
            out.write(white_image)

        # Mostrar en pantalla la imagen real con overlay del stickman y el canvas blanco
        cv2.imshow('Pose Detection', frame)
        cv2.imshow('Stickman with Trail', white_image)

        # Salir si se pulsa la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ---------------------------------------------------------------------------
# LIMPIEZA: asegurarnos de liberar recursos aunque ocurra una excepción
# ---------------------------------------------------------------------------
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# FIN DEL SCRIPT
# ---------------------------------------------------------------------------