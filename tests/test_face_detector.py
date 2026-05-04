import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from unittest.mock import Mock, patch, MagicMock
from detection.face_detector import FaceDetector


class TestFaceDetector(unittest.TestCase):
    """Тесты для детектора лиц"""
    
    def setUp(self):
        """Подготовка перед каждым тестом"""
        self.detector = FaceDetector()
        
        # Создаем тестовое изображение (черное)
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Добавляем "лицо" (прямоугольник светлого цвета)
        self.test_image[100:300, 150:250] = 200
    
    def test_detector_initialization(self):
        """Тест инициализации детектора"""
        self.assertIsNotNone(self.detector)
        self.assertTrue(hasattr(self.detector, '_face_cascade'))
    
    @patch('detection.face_detector.cv2')
    def test_filter_faces_valid_face(self, mock_cv2):
        """Тест фильтрации корректного лица"""
        faces = [(100, 100, 100, 120)]
        img_w, img_h = 640, 480
        
        filtered = self.detector._filter_faces(faces, img_w, img_h)
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], (100, 100, 100, 120))
    
    def test_filter_faces_invalid_aspect_ratio(self):
        """Тест фильтрации лица с неправильным соотношением сторон"""
        faces = [(100, 100, 200, 50)]
        img_w, img_h = 640, 480
        
        filtered = self.detector._filter_faces(faces, img_w, img_h)
        
        self.assertEqual(len(filtered), 0)
    
    def test_filter_faces_too_small(self):
        """Тест фильтрации слишком маленьких лиц"""
        faces = [(100, 100, 20, 25)]
        img_w, img_h = 640, 480
        
        filtered = self.detector._filter_faces(faces, img_w, img_h)
        
        self.assertEqual(len(filtered), 0)
    
    def test_filter_faces_near_edge(self):
        """Тест фильтрации лиц у края изображения"""
        faces = [(5, 100, 100, 120)]
        img_w, img_h = 640, 480
        
        filtered = self.detector._filter_faces(faces, img_w, img_h)
        
        self.assertEqual(len(filtered), 0)
    
    def test_detect_faces_empty_image(self):
        """Тест детекции на пустом изображении"""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = self.detector.detect_faces(empty_image)
        
        self.assertIsInstance(faces, list)


if __name__ == '__main__':
    unittest.main()