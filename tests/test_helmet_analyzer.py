import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import patch, MagicMock
from detection.helmet_analyzer import HelmetAnalyzer
import config


class TestHelmetAnalyzer(unittest.TestCase):
    """Тесты для анализатора касок"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.analyzer = HelmetAnalyzer()
        
        # Создаем тестовое изображение ROI
        self.test_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Создаем желтую область (имитация каски)
        self.yellow_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        # Желтый цвет в RGB
        self.yellow_roi[:, :] = [255, 255, 0]
    
    def test_analyze_helmet_none_roi(self):
        """Тест с None ROI"""
        has_helmet, score = self.analyzer.analyze_helmet(None)
        
        self.assertFalse(has_helmet)
        self.assertEqual(score, 0.0)
    
    def test_analyze_helmet_empty_roi(self):
        """Тест с пустым ROI"""
        empty_roi = np.array([])
        has_helmet, score = self.analyzer.analyze_helmet(empty_roi)
        
        self.assertFalse(has_helmet)
        self.assertEqual(score, 0.0)
    
    def test_analyze_helmet_too_small(self):
        """Тест с слишком маленьким ROI"""
        small_roi = np.zeros((5, 5, 3), dtype=np.uint8)
        has_helmet, score = self.analyzer.analyze_helmet(small_roi)
        
        self.assertFalse(has_helmet)
        self.assertEqual(score, 0.0)
    
    @patch('detection.helmet_analyzer.cv2')
    def test_analyze_with_hsv(self, mock_cv2):
        """Тест HSV анализа"""
        # Настраиваем моки для cv2
        mock_cv2.cvtColor.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.inRange.return_value = np.ones((100, 100), dtype=np.uint8) * 255
        mock_cv2.erode.return_value = np.ones((100, 100), dtype=np.uint8) * 255
        mock_cv2.getStructuringElement.return_value = np.ones((2, 2), np.uint8)
        
        has_helmet, score = self.analyzer._analyze_with_hsv(self.test_roi)
        
        self.assertTrue(has_helmet)
        self.assertGreater(score, 0)
    
    @patch('detection.helmet_analyzer.cv2')
    def test_analyze_with_hsv_exception(self, mock_cv2):
        """Тест обработки исключений в HSV анализе"""
        mock_cv2.cvtColor.side_effect = Exception("Test error")
        
        has_helmet, score = self.analyzer._analyze_with_hsv(self.test_roi)
        
        self.assertFalse(has_helmet)
        self.assertEqual(score, 0.0)
    
    def test_analyze_without_cv2(self):
        """Тест резервного алгоритма без OpenCV"""
        # Создаем яркое изображение
        bright_roi = np.ones((100, 100, 3), dtype=np.uint8) * 255
        
        has_helmet, score = self.analyzer._analyze_without_cv2(bright_roi)
        
        self.assertIsInstance(has_helmet, bool)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_hsv_color_ranges(self):
        """Тест корректности HSV диапазонов из конфига"""
        lower = np.array(config.HSV_YELLOW_LOWER)
        upper = np.array(config.HSV_YELLOW_UPPER)
        
        self.assertEqual(len(lower), 3)
        self.assertEqual(len(upper), 3)
        self.assertTrue((lower <= upper).all())


if __name__ == '__main__':
    unittest.main()