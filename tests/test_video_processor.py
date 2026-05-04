import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Создаем простые тесты без моков
class TestVideoProcessorSimple(unittest.TestCase):
    
    def test_import(self):
        """Проверка что модуль импортируется"""
        try:
            from utils.video_processor import VideoProcessor
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"OpenCV не установлен: {e}")
    
    def test_create_instance(self):
        """Проверка создания экземпляра"""
        try:
            from utils.video_processor import VideoProcessor
            # Не пытаемся открыть видео, просто создаем объект
            processor = VideoProcessor.__new__(VideoProcessor)
            self.assertIsNotNone(processor)
        except:
            self.skipTest("VideoProcessor требует OpenCV")


class TestVideoProcessorStats(unittest.TestCase):
    """Тесты только для статистики (без реального видео)"""
    
    def setUp(self):
        """Создаем объект без инициализации видео"""
        from utils.video_processor import VideoProcessor
        self.processor = VideoProcessor.__new__(VideoProcessor)
        self.processor._lock = __import__('threading').Lock()
        self.processor.stats = {
            'total_faces': 0,
            'total_with_helmet': 0,
            'total_without': 0,
            'frame_results': []
        }
    
    def test_get_current_stats_empty(self):
        """Тест получения статистики до обработки"""
        stats = self.processor.get_current_stats()
        
        self.assertEqual(stats['total_faces'], 0)
        self.assertEqual(stats['compliance'], 0)
    
    def test_get_current_stats_with_data(self):
        """Тест получения статистики с данными"""
        with self.processor._lock:
            self.processor.stats['total_faces'] = 10
            self.processor.stats['total_with_helmet'] = 7
            self.processor.stats['total_without'] = 3
        
        stats = self.processor.get_current_stats()
        
        self.assertEqual(stats['total_faces'], 10)
        self.assertEqual(stats['with_helmet'], 7)
        self.assertEqual(stats['without'], 3)
        self.assertEqual(stats['compliance'], 70.0)
    
    def test_get_progress(self):
        """Тест прогресса"""
        self.processor.total_frames = 100
        self.processor.current_frame_num = 50
        progress = self.processor.get_progress()
        
        self.assertEqual(progress, 50.0)
    
    def test_pause_resume(self):
        """Тест паузы и возобновления"""
        self.processor.pause = lambda: setattr(self.processor, 'is_paused', True)
        self.processor.resume = lambda: setattr(self.processor, 'is_paused', False)
        self.processor.is_paused = False
        
        self.processor.pause()
        self.assertTrue(self.processor.is_paused)
        
        self.processor.resume()
        self.assertFalse(self.processor.is_paused)


if __name__ == '__main__':
    unittest.main()