# detection/face_detector.py
import os
import numpy as np
import config

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

# MediaPipe отключен - не поддерживает Python 3.13
_MP_AVAILABLE = False


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
                print(f"Haar cascade загружен: {cascade_path}")
            else:
                print(f"Haar cascade не найден: {cascade_path}")
        except Exception as e:
            print(f"Ошибка загрузки Haar cascade: {e}")
            self._face_cascade = None
    
    def detect_faces(self, image_rgb):
        h_img, w_img = image_rgb.shape[:2]
        faces = []
        
        # Используем только Haar cascade (MediaPipe отключен)
        if self._face_cascade is not None:
            faces = self._detect_with_haar(image_rgb, w_img, h_img)
        
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
                    print(f"Лицо обнаружено: x={x}, y={y}, w={w}, h={h}")
        except Exception as e:
            print(f"Ошибка детекции Haar: {e}")
        return faces