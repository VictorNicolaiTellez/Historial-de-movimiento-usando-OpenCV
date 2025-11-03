# ---------------------------------------------------------------------------
# IMPORTACIONES
# ---------------------------------------------------------------------------
# OpenCV: captura de cámara, manipulación y dibujo sobre frames
import cv2
# MediaPipe: detección y seguimiento de poses humanas
import mediapipe as mp
# NumPy: creación de imágenes (canvas) y manejo de arrays
import numpy as np
# deque: estructura FIFO con límite para historial de posiciones (rastro)
from collections import deque

# ---------------------------------------------------------------------------
# CONFIGURACIÓN Y EXPLICACIÓN GLOBAL DEL CÓDIGO
# ---------------------------------------------------------------------------
# 1) MediaPipe Pose:
#    - mp.solutions.pose provee el modelo y utilidades para detectar 33 puntos
#      clave del cuerpo humano (landmarks).
#    - Cada landmark contiene coordenadas normalizadas (x, y, z) en rango 0..1
#      relativas al tamaño del frame. Para dibujar en píxeles hay que convertir
#      esas coordenadas multiplicando por el ancho/alto en píxeles del frame.
#    - min_detection_confidence: confianza mínima para considerar una detección
#      inicial válida. Valores típicos: 0.4-0.7.
#    - min_tracking_confidence: confianza mínima para mantener el seguimiento
#      entre frames (mejora estabilidad una vez detectado el cuerpo).
#
# 2) Flujo general del programa:
#    - Inicia la cámara y el modelo de MediaPipe.
#    - En un bucle infinito (hasta que se presione 'q' o falle la cámara):
#        a) captura un frame
#        b) procesa el frame con MediaPipe para obtener landmarks
#        c) si hay landmarks, los convierte a coordenadas píxel
#        d) guarda estas coordenadas en un historial limitado (deque)
#        e) dibuja en dos salidas visuales: la imagen real con el esqueleto y
#           una imagen blanca donde se representan el stickman y su rastro
#        f) escribe la imagen blanca en un archivo de vídeo
#    - Al terminar, libera todos los recursos (cámaras, ficheros, ventanas)
#
# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE MEDIAPIPE (detector/seguidor de pose)
# ---------------------------------------------------------------------------
# mp_pose: manejador para acceder a la solución de pose de MediaPipe
mp_pose = mp.solutions.pose
# Crear una instancia del modelo Pose con parámetros de confianza.
# Esta instancia hará la detección y el seguimiento de los 33 landmarks.
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------------------------------------------------------------------
# INICIALIZACIÓN DE LA CÁMARA
# ---------------------------------------------------------------------------
# Abrir la cámara por defecto (índice 0). Si tienes varias cámaras, cambia el
# índice (por ejemplo 1, 2...) o usa la ruta de un vídeo para pruebas.
cap = cv2.VideoCapture(0)

# Explicación :
#  - cap es un objeto que representa la cámara. Con cap.read() obtenemos un
#    frame (imagen) cada vez que lo llamamos.
#  - cap.get(propId) devuelve propiedades de la cámara; 3 = ancho, 4 = alto.

# Obtener ancho y alto del frame (en píxeles). Esto es útil para convertir
# coordenadas normalizadas (0..1) a coordenadas de píxeles (x*w, y*h).
frame_width = int(cap.get(3))   # CAP_PROP_FRAME_WIDTH
frame_height = int(cap.get(4))  # CAP_PROP_FRAME_HEIGHT

# ---------------------------------------------------------------------------
# CONFIGURAR VideoWriter para guardar el stickman en un archivo de vídeo
# ---------------------------------------------------------------------------
# VideoWriter requiere: nombre archivo, codec, fps, (ancho, alto)
# - El codec 'MJPG' (Motion-JPEG) suele funcionar en la mayoría de sistemas.
# - fps lo fijamos en 30, que es estándar. Si la cámara tiene otro fps, puedes
#   ajustarlo.
out = cv2.VideoWriter(
    'stickman_output.avi',                      # nombre archivo de salida
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), # codec
    30,                                         # frames por segundo
    (frame_width, frame_height)                 # tamaño de los frames a guardar
)

# ---------------------------------------------------------------------------
# ESTRUCTURAS PARA EL HISTORIAL Y LAS CONEXIONES DEL "STICKMAN"
# ---------------------------------------------------------------------------
# positions_history: cola con longitud máxima. Al usar maxlen=30
# mantenemos solo las últimas 30 posiciones para dibujar el rastro. Esto
# evita el uso ilimitado de memoria y limita la longitud del trazo visible.
positions_history = deque(maxlen=30)

# POSE_CONNECTIONS: lista de pares (tuplas) que definen qué puntos unir para
# formar las líneas del stickman. Los índices son los índices estándar de
# MediaPipe para los landmarks del cuerpo.
POSE_CONNECTIONS = [
    (11, 12), # hombros
    (11, 13), (13, 15), # brazo izquierdo: hombro->codo->muñeca
    (12, 14), (14, 16), # brazo derecho: hombro->codo->muñeca
    (11, 23), (12, 24), # torso: hombros->caderas
    (23, 24), # caderas (conexión entre cadera izquierda y derecha)
    (23, 25), (25, 27), # pierna izquierda: cadera->rodilla->tobillo
    (24, 26), (26, 28)  # pierna derecha: cadera->rodilla->tobillo
]

# ---------------------------------------------------------------------------
# BUCLE PRINCIPAL: captura, procesamiento y visualización
# ---------------------------------------------------------------------------
try:
    
    while cap.isOpened():
        # Capturar un frame de la cámara:
        # - ret: booleano que indica si la captura fue exitosa
        # - frame: la imagen capturada 
        ret, frame = cap.read()
        if not ret:
            # Si no se pudo leer el frame (cámara desconectada o fin de vídeo),
            # salimos del bucle.
            break

        # -------------------------------------------------------------------
        # PROCESAMIENTO CON MEDIAPIPE
        # -------------------------------------------------------------------
        # MediaPipe trabaja con imágenes en formato RGB; OpenCV usa BGR por
        # defecto, así que convertimos el espacio de color.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ejecutar el detector de pose sobre el frame convertido a RGB.
        # El resultado incluirá, si se detecta, un objeto pose_landmarks que
        # contiene los 33 puntos del cuerpo en coordenadas normalizadas.
        results = pose.process(image_rgb)

        # -------------------------------------------------------------------
        # PREPARAR CANVAS BLANCO DONDE DIBUJAR EL STICKMAN Y EL RASTRO
        # -------------------------------------------------------------------
        # Crear una imagen blanca del mismo tamaño que el frame capturado.
        white_image = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        # Si MediaPipe detectó landmarks (es decir, se ha encontrado una persona
        # y su pose fue reconocida), procedemos a convertirlos y dibujarlos.
        if results.pose_landmarks:
            # current_positions: lista de coordenadas en píxeles (x, y) para
            # todos los landmarks detectados en el frame actual.
            current_positions = []
            # Obtener dimensiones reales del frame para convertir coordenadas
            # normalizadas (0..1) a píxeles (x*w, y*h)
            h, w, _ = frame.shape

            # Recorrer cada landmark devuelto por MediaPipe y convertir a píxeles
            for landmark in results.pose_landmarks.landmark:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                current_positions.append((cx, cy))

            # Guardar las posiciones actuales en el historial. Si el
            # deque superó su longitud máxima, automáticamente descartará el
            # elemento más antiguo.
            positions_history.append(current_positions)

            # ----------------------------------------------------------------
            # DIBUJAR EL RASTRO (historial) EN LA IMAGEN BLANCA
            # ----------------------------------------------------------------
            # Recorremos todas las posiciones guardadas en el historial y
            # dibujamos las conexiones definidas en POSE_CONNECTIONS con un
            # color claro (amarillo). Esto produce el efecto de rastro.
            for prev_pos in positions_history:
                for connection in POSE_CONNECTIONS:
                    pt1 = prev_pos[connection[0]]
                    pt2 = prev_pos[connection[1]]
                    # cv2.line(img, punto1, punto2, color(B,G,R), grosor)
                    cv2.line(white_image, pt1, pt2, (0, 255, 255), 1)

            # ----------------------------------------------------------------
            # DIBUJAR EL STICKMAN ACTUAL  TANTO EN EL FRAME
            # ORIGINAL COMO EN LA IMAGEN BLANCA
            # ----------------------------------------------------------------
            # Dibujamos las mismas conexiones (pose actual) en rojo y marcamos
            # además las articulaciones con círculos pequeños para mayor claridad.
            for connection in POSE_CONNECTIONS:
                start_point = current_positions[connection[0]]
                end_point = current_positions[connection[1]]
                # Dibujar en la imagen de la cámara (para ver el overlay real)
                cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
                # Dibujar en la imagen blanca (para guardar en el vídeo)
                cv2.line(white_image, start_point, end_point, (0, 0, 255), 2)

            # Dibujar los puntos (articulaciones) como círculos rojos rellenos
            for point in current_positions:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
                cv2.circle(white_image, point, 4, (0, 0, 255), -1)

            # ----------------------------------------------------------------
            # GUARDAR EL FRAME DEL STICKMAN EN EL FICHERO DE SALIDA
            # ----------------------------------------------------------------
            # Escribimos la imagen blanca (que contiene el stickman y su rastro)
            # en el archivo de vídeo configurado anteriormente.
            out.write(white_image)

        # -------------------------------------------------------------------
        # MOSTRAR VENTANAS AL USUARIO
        # -------------------------------------------------------------------
        # Mostrar el frame original con overlay del esqueleto
        cv2.imshow('Pose Detection', frame)
        # Mostrar la imagen blanca con el stickman y su rastro
        cv2.imshow('Stickman with Trail', white_image)

        # Esperar 10 ms por la tecla 'q' para salir. La comprobación & 0xFF es
        # una práctica estándar para compatibilidad entre sistemas.
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ---------------------------------------------------------------------------
# FINALIZACIÓN: liberar recursos
# ---------------------------------------------------------------------------
finally:
    # Liberar la cámara
    cap.release()
    # Cerrar el archivo de vídeo
    out.release()
    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# FIN DEL PROGRAMA
# ---------------------------------------------------------------------------

