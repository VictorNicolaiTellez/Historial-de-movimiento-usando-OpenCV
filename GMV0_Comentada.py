# -*- coding: utf-8 -*-
# stickman_pose_record.py
# ---------------------------------------------------------------------------
# Programa: captura vídeo por cámara, detecta pose con MediaPipe Pose,
# dibuja un "stickman" simplificado en un canvas separado y guarda ese canvas
# como un archivo de vídeo (stickman_output.avi). Además muestra en pantalla
# el frame original con detección de movimiento y el stickman.
# ---------------------------------------------------------------------------

import cv2                      # OpenCV: captura vídeo, dibujo, lectura/escritura de archivos
import mediapipe as mp          # MediaPipe: detección y seguimiento de pose humana
import numpy as np              # Numpy: manejo eficiente de arrays y creación de canvas

# --- Inicialización de utilidades de MediaPipe ---------------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # (no usado directamente para dibujo en este script)

# --- Inicializar cámara ---------------------------------------------------
cap = cv2.VideoCapture(0)      # Abre la cámara por defecto (0). Cambiar si tienes varias cámaras.
ret, frame1 = cap.read()       # Leer primer frame
ret, frame2 = cap.read()       # Leer segundo frame (necesario para detectar movimiento por diferencia)
if not ret:
    # Si no se pudo leer la cámara, liberar recursos y terminar con error
    print("Error: no se pudo leer la cámara.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit(1)

# --- Configurar grabación del vídeo del stickman --------------------------
# Obtener dimensiones (ancho, alto) de los frames capturados por la cámara
ancho_video = int(cap.get(3))  # propiedad 3: CAP_PROP_FRAME_WIDTH
alto_video = int(cap.get(4))   # propiedad 4: CAP_PROP_FRAME_HEIGHT
fps = 30                       # frames por segundo objetivo para la grabación (usado al crear VideoWriter)

# Crear objeto VideoWriter para guardar el canvas del stickman en disco.
# Parámetros: nombre archivo, codec (MJPG), fps, (ancho, alto)
# Nota: el tamaño del canvas que grabamos debe coincidir con (ancho_video, alto_video).
salida_video = out = cv2.VideoWriter(
    'stickman_output.avi',
    cv2.VideoWriter_fourcc('M','J','P','G'),
    30,
    (ancho_video, alto_video)
)

# --- Diccionario para almacenar las trayectorias de cada articulación ----
# La clave será el índice del landmark de MediaPipe (p. ej. 11 = hombro izquierdo),
# y el valor será una lista de coordenadas (x,y) en el canvas que representen la trayectoria.
trayectorias = {}

# ------------------- Función para dibujar un stickman ---------------------
def draw_stickman_from_landmarks(landmarks, canvas_size=(400, 600)):
    """
    Dibuja un stickman simplificado a partir de los 'landmarks' (puntos normalizados)
    devueltos por MediaPipe Pose.

    Parámetros:
      - landmarks: lista de objetos Landmark de MediaPipe (cada uno tiene .x y .y normalizados 0..1)
                   o None si no hay detección.
      - canvas_size: tupla (ancho, alto) del canvas donde se dibujará el stickman.
                     Por defecto (400, 600). En el script principal se pasa (ancho_video, alto_video)
                     para que coincida con la resolución de salida.

    Retorna:
      - canvas: imagen (numpy array) con fondo blanco y el stickman dibujado.
    """
    canvas_w, canvas_h = canvas_size
    # Crear fondo blanco (3 canales BGR) del tamaño solicitado
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    if landmarks is None:
        # Si no hay landmarks, devolvemos solo el canvas en blanco
        return canvas

    # Convertir landmarks (normalizados) a lista sencilla de coordenadas (x, y)
    pts = [(lm.x, lm.y) for lm in landmarks]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    # Calcular bounding box de los puntos (en coordenadas normalizadas)
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    # Evitar división por cero si las coordenadas son todas iguales
    w = maxx - minx if maxx > minx else 1e-6
    h = maxy - miny if maxy > miny else 1e-6

    # Escala para ajustar la figura al canvas: se usa un factor que mantenga la relación
    # y deje márgenes (0.7 y 0.8 son factores heurísticos)
    scale = min((canvas_w * 0.7) / w, (canvas_h * 0.8) / h)

    # Centro del bounding box (en coordenadas normalizadas)
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    # Función local que transforma coordenadas normalizadas (xn, yn) -> coordenadas en píxeles del canvas
    def proj(xn, yn):
        x = int((xn - cx) * scale + canvas_w / 2)
        y = int((yn - cy) * scale + canvas_h / 2)
        return x, y

    # Lista de índices de MediaPipe Pose que se usan para el stickman (selección reducida)
    # 0 = nariz, 11 = hombro izquierdo, 12 = hombro derecho, 13 = codo izquierdo, 14 = codo derecho, etc.
    indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

    # Diccionario P que guardará las coordenadas proyectadas en píxeles para cada índice
    P = {}
    for idx in indices:
        lm = landmarks[idx]              # obtener landmark (con .x, .y)
        P[idx] = proj(lm.x, lm.y)       # proyectar a coordenadas de canvas

        # --- Almacenar trayectoria de cada punto en el diccionario global 'trayectorias'
        if idx not in trayectorias:
            trayectorias[idx] = []
        trayectorias[idx].append(P[idx])

        # Mantener solo las últimas N posiciones para no crecer indefinidamente (aquí N=50)
        if len(trayectorias[idx]) > 50:
            trayectorias[idx].pop(0)

    # --- Dibujar trayectorias (líneas grises leves) para cada articulación -----------
    for idx, trazo in trayectorias.items():
        for i in range(1, len(trazo)):
            # Dibujamos línea entre puntos consecutivos de la trayectoria
            cv2.line(canvas, trazo[i - 1], trazo[i], (180, 180, 180), 2)

    # --- Dibujar conexiones del stickman (lista 'conexiones' define pares de índices a unir)
    thickness = 4
    color = (0, 0, 0)  # negro

    conexiones = [
        (11, 12), (11, 23), (12, 24), (23, 24),  # torso / cintura
        (11, 13), (13, 15), (12, 14), (14, 16),  # brazos (hombro->codo->muñeca)
        (23, 25), (25, 27), (24, 26), (26, 28)   # piernas (cadera->rodilla->tobillo)
    ]

    for (a, b) in conexiones:
        if a in P and b in P:
            cv2.line(canvas, P[a], P[b], color, thickness)

    # --- Dibujar puntos de articulación (pequeños círculos)
    for p in P.values():
        cv2.circle(canvas, p, 5, (50, 50, 200), -1)  # círculo relleno azul-oscuro

    return canvas

# ---------------- Inicializar MediaPipe Pose y bucle principal -------------
# Usamos 'with' para asegurar que la sesión se libera correctamente al salir.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        # ------------------ DETECCIÓN DE MOVIMIENTO (por diferencia de frames) --------------
        # Se calcula la diferencia absoluta entre frame1 y frame2 para detectar movimiento.
        diff = cv2.absdiff(frame1, frame2)

        # Convertir a escala de grises (más barato computacionalmente)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Aplicar desenfoque para eliminar ruido pequeño
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Umbralizar para obtener una máscara binaria de movimiento
        _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)

        # Dilatar para rellenar huecos y agrupar regiones
        dilated = cv2.dilate(thresh, None, iterations=3)

        # Encontrar contornos en la máscara de movimiento
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Copia del frame para dibujar resultados
        frame_out = frame1.copy()
        h_full, w_full = frame_out.shape[:2]

        # ------------------ PROCESAR POSE (en el frame entero) -----------------------------
        # MediaPipe espera imágenes en RGB
        frame_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        results_full = pose.process(frame_rgb)  # procesar para obtener landmarks de pose

        if results_full.pose_landmarks:
            # Si hay landmarks, dibujamos las conexiones entre cada par definido por MediaPipe
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start_lm = results_full.pose_landmarks.landmark[start_idx]
                end_lm = results_full.pose_landmarks.landmark[end_idx]

                # Convertir coordenadas normalizadas a píxeles en el frame original
                x1 = int(start_lm.x * w_full)
                y1 = int(start_lm.y * h_full)
                x2 = int(end_lm.x * w_full)
                y2 = int(end_lm.y * h_full)

                # Dibujar línea entre las dos articulaciones
                cv2.line(frame_out, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Dibujar un círculo en cada landmark detectado
            for lm in results_full.pose_landmarks.landmark:
                abs_x = int(lm.x * w_full)
                abs_y = int(lm.y * h_full)
                cv2.circle(frame_out, (abs_x, abs_y), 4, (0, 128, 255), -1)

        # ------------------ DIBUJAR RECTÁNGULOS DE MOVIMIENTO -----------------------------
        # Recorremos contornos encontrados y dibujamos rectángulos si el área es suficientemente grande
        for contour in contours:
            if cv2.contourArea(contour) < 1000:  # filtro de ruido: ignorar áreas pequeñas
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ------------------ GENERAR Y MOSTRAR EL CANVAS DEL STICKMAN ----------------------
        # Si pose_landmarks existe, pasamos la lista de landmarks; si no, pasamos None.
        stickman_canvas = draw_stickman_from_landmarks(
            results_full.pose_landmarks.landmark if results_full.pose_landmarks else None,
            canvas_size=(ancho_video, alto_video)
        )

        # Mostrar la ventana con el stickman y la ventana con el frame original+detección
        cv2.imshow("Stickman Model", stickman_canvas)
        cv2.imshow("Movimiento + Grabacion ", frame_out)

        # --- GUARDAR el frame del stickman en el archivo de vídeo
        salida_video.write(stickman_canvas)

        # --- AVANZAR FRAMES para la detección de movimiento en la siguiente iteración
        frame1 = frame2
        ret, frame2 = cap.read()
        if not ret:
            # Si no se pudieron leer más frames, salimos del bucle
            break

        # Esperar ~40 ms por tecla (aprox 25 fps), si se presiona ESC (27) salimos
        if cv2.waitKey(40) == 27:
            break

# ------------------------ LIBERAR RECURSOS --------------------------------
cap.release()
salida_video.release()
cv2.destroyAllWindows()
print("Video guardado como 'stickman_output.avi'")
