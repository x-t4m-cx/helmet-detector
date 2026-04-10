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
        if roi_rgb is None or roi_rgb.size == 0:
            return False, 0.0
        
        if not _CV2_AVAILABLE:
            return self._analyze_without_cv2(roi_rgb)
        
        return self._analyze_with_hsv(roi_rgb)
    
    def _analyze_with_hsv(self, roi_rgb):
        try:
            hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
            lower = np.array(config.HSV_YELLOW_LOWER)
            upper = np.array(config.HSV_YELLOW_UPPER)
            mask = cv2.inRange(hsv, lower, upper)
            ratio = mask.mean() / 255.0
            has_helmet = ratio > config.HELMET_COLOR_RATIO_THRESHOLD
            score = min(1.0, ratio * config.HELMET_SCORE_MULTIPLIER)
            return bool(has_helmet), float(score)
        except Exception:
            return False, 0.0
    
    def _analyze_without_cv2(self, roi_rgb):
        arr = roi_rgb.astype(np.float32) / 255.0
        v = np.max(arr, axis=2)
        mask_bright = (v > 0.85)
        helmet_pixels = np.count_nonzero(mask_bright)
        total = roi_rgb.shape[0] * roi_rgb.shape[1]
        ratio = helmet_pixels / total if total else 0.0
        has_helmet = ratio > config.HELMET_COLOR_RATIO_THRESHOLD
        score = min(1.0, ratio * config.HELMET_SCORE_MULTIPLIER)
        return bool(has_helmet), float(score)