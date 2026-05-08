import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import patch, MagicMock
from utils.image_utils import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    """Тесты для утилит обработки изображений"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.processor = ImageProcessor()
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_enhance_contrast_no_cv2(self):
        """Тест улучшения контраста без OpenCV"""
        with patch('utils.image_utils._CV2_AVAILABLE', False):
            result = self.processor.enhance_contrast(self.test_image)
            
            # Должен вернуть оригинальное изображение
            np.testing.assert_array_equal(result, self.test_image)
    
    @patch('utils.image_utils.cv2')
    def test_enhance_contrast_with_cv2(self, mock_cv2):
        """Тест улучшения контраста с OpenCV"""
        # Настраиваем моки
        mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cv2.split.return_value = (np.zeros((480, 640), dtype=np.uint8), 
                                       np.zeros((480, 640), dtype=np.uint8), 
                                       np.zeros((480, 640), dtype=np.uint8))
        mock_cv2.createCLAHE.return_value = MagicMock()
        mock_cv2.createCLAHE.return_value.apply.return_value = np.zeros((480, 640), dtype=np.uint8)
        mock_cv2.merge.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = self.processor.enhance_contrast(self.test_image)
        
        self.assertIsNotNone(result)
        mock_cv2.cvtColor.assert_called()
    
    @patch('utils.image_utils.cv2')
    def test_enhance_contrast_exception(self, mock_cv2):
        """Тест обработки исключений при улучшении контраста"""
        mock_cv2.cvtColor.side_effect = Exception("Test error")
        
        result = self.processor.enhance_contrast(self.test_image)
        
        # Должен вернуть оригинал при ошибке
        np.testing.assert_array_equal(result, self.test_image)
    
    def test_get_helmet_roi_basic(self):
        """Тест получения ROI для каски (базовый случай)"""
        x, y, w, h = 100, 100, 80, 100
        img_w, img_h = 640, 480
        
        roi = self.processor.get_helmet_roi(x, y, w, h, img_w, img_h)
        
        x1, y1, x2, y2 = roi
        
        # Проверяем границы
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, img_w)
        self.assertLessEqual(y2, img_h)
        
        # Проверяем размеры
        self.assertGreater(x2 - x1, 0)
        self.assertGreater(y2 - y1, 0)
    
    def test_get_helmet_roi_near_edges(self):
        """Тест получения ROI у краев изображения"""
        x, y, w, h = 10, 10, 80, 100
        img_w, img_h = 640, 480
        
        roi = self.processor.get_helmet_roi(x, y, w, h, img_w, img_h)
        
        x1, y1, x2, y2 = roi
        
        # ROI не должен выходить за границы
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, img_w)
        self.assertLessEqual(y2, img_h)
    
    def test_get_stability_key(self):
        """Тест получения ключа стабильности"""
        x, y, w, h = 123, 456, 78, 90
        
        key = self.processor.get_stability_key(x, y, w, h)
        
        # Координаты должны быть округлены до десятков
        self.assertEqual(key[0], 120)  # 123 -> 120
        self.assertEqual(key[1], 460)  # 456 -> 460
        self.assertEqual(key[2], 80)   # 78 -> 80
        self.assertEqual(key[3], 90)   # 90 -> 90
    
    def test_get_stability_key_negative(self):
        """Тест получения ключа стабильности с отрицательными координатами"""
        x, y, w, h = -5, -15, 78, 90
        
        key = self.processor.get_stability_key(x, y, w, h)
        
        # Даже отрицательные должны корректно округляться
        self.assertIsInstance(key, tuple)
        self.assertEqual(len(key), 4)


if __name__ == '__main__':
    unittest.main()