# utils/image_utils.py
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class ImageProcessor:
    @staticmethod
    def enhance_contrast(image_rgb):
        if not _CV2_AVAILABLE:
            return image_rgb
        
        try:
            ycc = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycc)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            ycc = cv2.merge([y, cr, cb])
            return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
        except Exception:
            return image_rgb
    
    @staticmethod
    def get_helmet_roi(x, y, w, h, img_w, img_h):
        k_up = int(h * 0.95)
        helmet_y1 = max(0, y - k_up // 2)
        helmet_y2 = max(0, y + int(h * 0.08))
        helmet_x1 = max(0, x)
        helmet_x2 = min(img_w, x + w)
        return (helmet_x1, helmet_y1, helmet_x2, helmet_y2)
    
    @staticmethod
    def get_stability_key(x, y, w, h):
        return (
            round(x / 10) * 10,
            round(y / 10) * 10,
            round(w / 10) * 10,
            round(h / 10) * 10
        )