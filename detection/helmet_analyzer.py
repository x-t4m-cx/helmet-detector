# detection/helmet_analyzer.py
import numpy as np
import config

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class HelmetAnalyzer:

    def analyze_helmet(self, roi_rgb):
        """Анализирует ROI на наличие каски"""
        if roi_rgb is None or roi_rgb.size == 0:
            return False, 0.0

        # Проверка минимального размера ROI
        h, w = roi_rgb.shape[:2]
        if h < 10 or w < 10:
            return False, 0.0

        if not _CV2_AVAILABLE:
            return self._analyze_without_cv2(roi_rgb)

        return self._analyze_with_hsv(roi_rgb)

    def _analyze_with_hsv(self, roi_rgb):
        """Анализ с использованием HSV (проверенный рабочий вариант)"""
        try:
            hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)

            lower = np.array(config.HSV_YELLOW_LOWER)
            upper = np.array(config.HSV_YELLOW_UPPER)
            mask = cv2.inRange(hsv, lower, upper)

            # Простая очистка от шума (только эрозия, без дилатации)
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

            ratio = mask.mean() / 255.0
            has_helmet = ratio > config.HELMET_COLOR_RATIO_THRESHOLD
            score = min(1.0, ratio * config.HELMET_SCORE_MULTIPLIER)

            # Логирование для отладки (можно убрать потом)
            if has_helmet and score > 0.3:
                print(f"Каска: ratio={ratio:.2f}, score={score:.2f}")

            return bool(has_helmet), float(score)

        except Exception as e:
            print(f"Ошибка анализа HSV: {e}")
            return False, 0.0

    def _analyze_without_cv2(self, roi_rgb):
        """Резервный алгоритм"""
        arr = roi_rgb.astype(np.float32) / 255.0
        v = np.max(arr, axis=2)
        mask_bright = (v > 0.85)
        ratio = mask_bright.mean()
        has_helmet = ratio > config.HELMET_COLOR_RATIO_THRESHOLD
        score = min(1.0, ratio * config.HELMET_SCORE_MULTIPLIER)
        return bool(has_helmet), float(score)