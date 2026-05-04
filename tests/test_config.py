import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


class TestConfig(unittest.TestCase):
    """Тесты для конфигурации"""
    
    def test_face_detection_params_exist(self):
        """Проверка наличия параметров детекции лиц"""
        self.assertTrue(hasattr(config, 'MIN_FACE_SIZE_RATIO'))
        self.assertTrue(hasattr(config, 'MAX_FACE_SIZE_RATIO'))
        self.assertTrue(hasattr(config, 'MIN_FACE_Y_RATIO'))
        self.assertTrue(hasattr(config, 'MAX_FACE_Y_RATIO'))
        self.assertTrue(hasattr(config, 'FACE_ASPECT_MIN'))
        self.assertTrue(hasattr(config, 'FACE_ASPECT_MAX'))
    
    def test_face_detection_params_valid(self):
        """Проверка валидности параметров детекции лиц"""
        self.assertGreater(config.MIN_FACE_SIZE_RATIO, 0)
        self.assertLess(config.MIN_FACE_SIZE_RATIO, 1)
        self.assertGreater(config.MAX_FACE_SIZE_RATIO, config.MIN_FACE_SIZE_RATIO)
        self.assertLess(config.MAX_FACE_SIZE_RATIO, 1)
        
        self.assertGreater(config.FACE_ASPECT_MIN, 0)
        self.assertGreater(config.FACE_ASPECT_MAX, config.FACE_ASPECT_MIN)
    
    def test_haar_params_valid(self):
        """Проверка валидности параметров Haar cascade"""
        self.assertTrue(hasattr(config, 'HAAR_SCALE_FACTOR'))
        self.assertTrue(hasattr(config, 'HAAR_MIN_NEIGHBORS'))
        
        self.assertGreater(config.HAAR_SCALE_FACTOR, 1.0)
        self.assertLess(config.HAAR_SCALE_FACTOR, 1.2)
        self.assertGreater(config.HAAR_MIN_NEIGHBORS, 0)
    
    def test_helmet_params_exist(self):
        """Проверка наличия параметров для каски"""
        self.assertTrue(hasattr(config, 'HELMET_COLOR_RATIO_THRESHOLD'))
        self.assertTrue(hasattr(config, 'HELMET_SCORE_MULTIPLIER'))
        self.assertTrue(hasattr(config, 'HSV_YELLOW_LOWER'))
        self.assertTrue(hasattr(config, 'HSV_YELLOW_UPPER'))
    
    def test_helmet_params_valid(self):
        """Проверка валидности параметров для каски"""
        self.assertGreater(config.HELMET_COLOR_RATIO_THRESHOLD, 0)
        self.assertLess(config.HELMET_COLOR_RATIO_THRESHOLD, 1)
        
        self.assertGreater(config.HELMET_SCORE_MULTIPLIER, 0)
        
        # Проверка HSV диапазонов
        self.assertEqual(len(config.HSV_YELLOW_LOWER), 3)
        self.assertEqual(len(config.HSV_YELLOW_UPPER), 3)
        
        # H (оттенок) должен быть в диапазоне 0-180 для OpenCV
        self.assertGreaterEqual(config.HSV_YELLOW_LOWER[0], 0)
        self.assertLessEqual(config.HSV_YELLOW_UPPER[0], 180)
        
        # S и V в диапазоне 0-255
        self.assertGreaterEqual(config.HSV_YELLOW_LOWER[1], 0)
        self.assertLessEqual(config.HSV_YELLOW_UPPER[1], 255)
        self.assertGreaterEqual(config.HSV_YELLOW_LOWER[2], 0)
        self.assertLessEqual(config.HSV_YELLOW_UPPER[2], 255)
    
    def test_video_params_exist(self):
        """Проверка наличия параметров видео"""
        self.assertTrue(hasattr(config, 'VIDEO_PROCESS_EVERY_N_FRAMES'))
        self.assertTrue(hasattr(config, 'VIDEO_BUFFER_SIZE'))
        self.assertTrue(hasattr(config, 'VIDEO_UPDATE_INTERVAL_MS'))
    
    def test_video_params_valid(self):
        """Проверка валидности параметров видео"""
        self.assertGreater(config.VIDEO_PROCESS_EVERY_N_FRAMES, 0)
        self.assertGreater(config.VIDEO_BUFFER_SIZE, 0)
        self.assertGreater(config.VIDEO_UPDATE_INTERVAL_MS, 0)
    
    def test_camera_params_exist(self):
        """Проверка наличия параметров камеры"""
        self.assertTrue(hasattr(config, 'MAX_CAMERAS_TO_CHECK'))
        self.assertTrue(hasattr(config, 'CAMERA_FRAME_DELAY'))
        self.assertTrue(hasattr(config, 'USE_DSHOW_ON_WINDOWS'))


if __name__ == '__main__':
    unittest.main()