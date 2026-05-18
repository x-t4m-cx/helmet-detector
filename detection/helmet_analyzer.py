# detection/helmet_analyzer.py
import numpy as np
import logging
import config

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

logger = logging.getLogger(__name__)


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

            # Улучшенная очистка маски: убираем шум и закрываем мелкие пробелы.
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            ratio = mask.mean() / 255.0
            has_helmet, score, helmet_score = self._build_decision(ratio)

            # Логирование для отладки (можно убрать потом)
            if score > 0.3:
                logger.debug(
                    "Результат каски: has_helmet=%s ratio=%.3f helmet_score=%.3f confidence=%.3f",
                    has_helmet, ratio, helmet_score, score
                )

            return bool(has_helmet), float(score)

        except Exception:
            logger.exception("Ошибка анализа HSV")
            return False, 0.0

    def _analyze_without_cv2(self, roi_rgb):
        """Резервный алгоритм"""
        arr = roi_rgb.astype(np.float32) / 255.0
        v = np.max(arr, axis=2)
        mask_bright = (v > 0.85)
        ratio = mask_bright.mean()
        has_helmet, score, _ = self._build_decision(ratio)
        return bool(has_helmet), float(score)

    def _build_decision(self, ratio):
        """
        Формирует решение по каске и confidence.
        helmet_score: вероятность "есть каска" (0..1)
        score: уверенность в итоговой классификации (0..1), в т.ч. для "без каски".
        """
        helmet_score = min(1.0, ratio * config.HELMET_SCORE_MULTIPLIER)
        has_helmet = helmet_score >= config.HELMET_COLOR_RATIO_THRESHOLD
        score = helmet_score if has_helmet else (1.0 - helmet_score)
        return bool(has_helmet), float(score), float(helmet_score)