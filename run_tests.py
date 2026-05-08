#!/usr/bin/env python3
"""
Запуск всех юнит-тестов для проекта распознавания лиц
Использование: python run_tests.py
"""

import unittest
import sys
import os

# Добавляем корневую директорию проекта в PATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all_tests():
    """Запускает все тесты и возвращает результат"""
    
    # Загружаем все тесты из директории tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Запускаем тесты с подробным выводом
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Выводим сводку
    print("\n" + "="*60)
    print("СВОДКА ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"Выполнено тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Ошибок: {len(result.errors)}")
    print(f"Провалов: {len(result.failures)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Создаем директорию для тестов, если её нет
    os.makedirs('helmet-detector-main/tests', exist_ok=True)
    
    # Создаем __init__.py в директории tests, если его нет
    init_file = os.path.join('helmet-detector-main/tests', '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# Тесты для проекта распознавания лиц\n")
    
    success = run_all_tests()
    
    # Возвращаем код завершения (0 для успеха, 1 для ошибки)
    sys.exit(0 if success else 1)