# detection/face_detector.py
import os
import logging
import numpy as np
import config

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# MediaPipe отключен - не поддерживает Python 3.13
_MP_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self):
        self._face_cascade = None
        if _CV2_AVAILABLE:
            self._init_haar_cascade()

    def _init_haar_cascade(self):
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.isfile(cascade_path):
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("Haar cascade загружен: %s", cascade_path)
            else:
                logger.warning("Haar cascade не найден: %s", cascade_path)
        except Exception as e:
            logger.exception("Ошибка загрузки Haar cascade")
            self._face_cascade = None

    def _filter_faces(self, faces, image_rgb, img_w, img_h):
        """Фильтрует ложные срабатывания детектора лиц"""
        filtered = []
        gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY) if _CV2_AVAILABLE else None

        for (x, y, w, h) in faces:
            # 1. Проверка соотношения сторон (лицо не должно быть слишком широким/узким)
            aspect = w / float(h)
            if not (config.FACE_ASPECT_MIN < aspect < config.FACE_ASPECT_MAX):
                continue

            # 2. Проверка размера относительно кадра
            face_size_ratio = max(w / float(img_w), h / float(img_h))
            if not (config.MIN_FACE_SIZE_RATIO <= face_size_ratio <= config.MAX_FACE_SIZE_RATIO):
                continue

            # 3. Проверка вертикальной позиции
            y_ratio = y / float(img_h)
            if not (config.MIN_FACE_Y_RATIO <= y_ratio <= config.MAX_FACE_Y_RATIO):
                continue

            # 4. Проверка структуры ROI: слишком "плоские" и без контуров области отсекаем
            if gray_full is not None:
                roi = gray_full[y:y + h, x:x + w]
                if roi.size == 0:
                    continue

                roi_std = float(np.std(roi))
                if roi_std < config.MIN_FACE_STD_DEV:
                    continue

                edges = cv2.Canny(roi, threshold1=60, threshold2=140)
                edge_density = float(np.count_nonzero(edges)) / float(edges.size)
                if edge_density < config.MIN_FACE_EDGE_DENSITY:
                    continue

            filtered.append((x, y, w, h))

        return filtered

    def detect_faces(self, image_rgb):
        h_img, w_img = image_rgb.shape[:2]
        faces = []

        if self._face_cascade is not None:
            faces = self._detect_with_haar(image_rgb, w_img, h_img)
            # Добавляем фильтрацию
            faces = self._filter_faces(faces, image_rgb, w_img, h_img)

        return faces

    def _detect_with_haar(self, image_rgb, w_img, h_img):
        faces = []
        try:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            min_size = (max(30, w_img // 30), max(30, h_img // 30))
            dets = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=config.HAAR_SCALE_FACTOR,
                minNeighbors=config.HAAR_MIN_NEIGHBORS,
                minSize=min_size
            )
            for (x, y, w, h) in dets:
                aspect = w / float(h)
                if 0.2 < aspect < 1.5:
                    faces.append((x, y, w, h))
                    logger.debug("Лицо обнаружено: x=%s, y=%s, w=%s, h=%s", x, y, w, h)
        except Exception as e:
            logger.exception("Ошибка детекции Haar")
        return faces