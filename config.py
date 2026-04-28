# config.py
import os

# Пороги для детекции лиц
MIN_FACE_SIZE_RATIO = 0.08
MAX_FACE_SIZE_RATIO = 0.6
MIN_FACE_Y_RATIO = 0.05
MAX_FACE_Y_RATIO = 0.90
FACE_ASPECT_MIN = 0.5
FACE_ASPECT_MAX = 1.5

# Пороги для Haar cascade (увеличиваем чувствительность)
HAAR_SCALE_FACTOR = 1.08      # Между 1.05 и 1.1
HAAR_MIN_NEIGHBORS = 3       # Увеличен с 3 (требует больше подтверждений)

# Пороги для анализа каски
HELMET_COLOR_RATIO_THRESHOLD = 0.15   # Вернули с 0.25 (0.18 был рабочий)
HELMET_SCORE_MULTIPLIER = 2.5      # Уменьшен с 2.5

# Стабилизация для камеры
STABILITY_REQUIRED_FRAMES = 3

# Цветовые диапазоны HSV для каски (жёлтый)
HSV_YELLOW_LOWER = (15, 80, 80)    # Было (15, 80, 80) - более строгий
HSV_YELLOW_UPPER = (35, 255, 255)    # Было (35, 255, 255)

# Путь к шрифту
FONT_PATH = "arial.ttf"

# Настройки камеры
MAX_CAMERAS_TO_CHECK = 5
CAMERA_FRAME_DELAY = 0.03
USE_DSHOW_ON_WINDOWS = True

#видео
VIDEO_PROCESS_EVERY_N_FRAMES = 2  # Обрабатывать каждый N-ый кадр
VIDEO_BUFFER_SIZE = 30            # Размер буфера кадров
VIDEO_UPDATE_INTERVAL_MS = 33     # Интервал обновления UI (~30 FPS)