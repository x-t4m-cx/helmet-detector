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
HAAR_SCALE_FACTOR = 1.05  # Уменьшен для лучшей детекции
HAAR_MIN_NEIGHBORS = 3     # Уменьшен для лучшей детекции

# Пороги для анализа каски
HELMET_COLOR_RATIO_THRESHOLD = 0.18
HELMET_SCORE_MULTIPLIER = 2.5

# Стабилизация для камеры
STABILITY_REQUIRED_FRAMES = 3

# Цветовые диапазоны HSV для каски (жёлтый)
HSV_YELLOW_LOWER = (15, 80, 80)
HSV_YELLOW_UPPER = (35, 255, 255)

# Путь к шрифту
FONT_PATH = "arial.ttf"

# Настройки камеры
MAX_CAMERAS_TO_CHECK = 5
CAMERA_FRAME_DELAY = 0.03
USE_DSHOW_ON_WINDOWS = True