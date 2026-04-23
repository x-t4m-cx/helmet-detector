# utils/video_processor.py
import threading
import queue
import time
import numpy as np
from PIL import Image

try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    print("Предупреждение: OpenCV (cv2) не установлен. Видеофункции будут недоступны.")


class VideoProcessor:
    """Оптимизированный процессор для видео с буферизацией кадров"""

    def __init__(self, video_path=None, max_buffer_size=30, process_every_n_frames=2):
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) не установлен. Установите: pip install opencv-python")

        self.video_path = video_path
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.current_frame = None
        self.frame_queue = queue.Queue(maxsize=max_buffer_size)
        self.result_queue = queue.Queue()
        self.thread_read = None
        self.thread_process = None
        self.fps = 0
        self.total_frames = 0
        self.current_frame_num = 0
        self.process_every_n_frames = process_every_n_frames
        self.frame_skip_counter = 0
        self._lock = threading.Lock()

        # Кэш для результатов
        self.results_cache = {}
        self.cache_max_size = 100

        # Статистика
        self.stats = {
            'total_faces': 0,
            'total_with_helmet': 0,
            'total_without': 0,
            'frame_results': []
        }

    def open_video(self, video_path):
        """Открывает видеофайл"""
        if not _CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) не установлен. Установите: pip install opencv-python")

        self.close()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        # Получаем информацию о видео
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame_num = 0

        return {
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration': self.total_frames / self.fps if self.fps > 0 else 0,
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

    def start_playback(self, process_callback=None):
        """Запускает воспроизведение и обработку"""
        if self.cap is None:
            return False

        self.is_playing = True
        self.is_paused = False
        self.stats = {
            'total_faces': 0,
            'total_with_helmet': 0,
            'total_without': 0,
            'frame_results': []
        }

        # Поток для чтения кадров
        self.thread_read = threading.Thread(target=self._read_frames, daemon=True)
        self.thread_read.start()

        # Поток для обработки
        if process_callback:
            self.thread_process = threading.Thread(
                target=self._process_frames,
                args=(process_callback,),
                daemon=True
            )
            self.thread_process.start()

        return True

    def _read_frames(self):
        """Читает кадры из видео в очередь"""
        while self.is_playing:
            if self.is_paused:
                time.sleep(0.05)
                continue

            if self.cap is None or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()

            if not ret:
                # Видео закончилось
                self.is_playing = False
                break

            self.current_frame_num += 1

            # Конвертируем в RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Пытаемся добавить в очередь (не блокируем)
            try:
                self.frame_queue.put_nowait({
                    'frame_num': self.current_frame_num,
                    'image': rgb_frame,
                    'timestamp': self.current_frame_num / self.fps if self.fps > 0 else 0
                })
            except queue.Full:
                # Очередь переполнена - пропускаем кадр
                pass

            # Контроль скорости воспроизведения
            if self.fps > 0:
                time.sleep(1.0 / self.fps)

        # Сигнализируем о завершении
        self.frame_queue.put(None)

    def _process_frames(self, callback):
        """Обрабатывает кадры из очереди"""
        while self.is_playing:
            try:
                # Ждем кадр с таймаутом
                frame_data = self.frame_queue.get(timeout=0.5)

                if frame_data is None:
                    break

                # Пропускаем кадры для оптимизации
                self.frame_skip_counter += 1
                if self.frame_skip_counter % self.process_every_n_frames != 0:
                    continue

                # Вызываем callback для обработки
                result = callback(frame_data['image'], frame_data['frame_num'])

                if result:
                    # Обновляем статистику
                    with self._lock:
                        faces_count = len(result)
                        with_helmet = sum(1 for r in result if r.get('has_helmet', False))
                        without = faces_count - with_helmet

                        self.stats['total_faces'] += faces_count
                        self.stats['total_with_helmet'] += with_helmet
                        self.stats['total_without'] += without
                        self.stats['frame_results'].append({
                            'frame_num': frame_data['frame_num'],
                            'timestamp': frame_data['timestamp'],
                            'faces': faces_count,
                            'with_helmet': with_helmet,
                            'without': without,
                            'results': result
                        })

                        # Ограничиваем размер кэша
                        if len(self.stats['frame_results']) > self.cache_max_size:
                            self.stats['frame_results'].pop(0)

                    # Отправляем результат
                    try:
                        self.result_queue.put_nowait({
                            'frame_num': frame_data['frame_num'],
                            'image': frame_data['image'],
                            'timestamp': frame_data['timestamp'],
                            'results': result,
                            'stats': self.get_current_stats()
                        })
                    except queue.Full:
                        pass

            except queue.Empty:
                continue

    def get_next_result(self, timeout=0.033):
        """Получает следующий результат обработки"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pause(self):
        """Пауза"""
        self.is_paused = True

    def resume(self):
        """Возобновление"""
        self.is_paused = False

    def seek(self, frame_percent):
        """Перемотка (0-100%)"""
        if self.cap is None:
            return

        target_frame = int(self.total_frames * frame_percent / 100)
        target_frame = max(0, min(target_frame, self.total_frames - 1))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.current_frame_num = target_frame

        # Очищаем очереди при перемотке
        self._clear_queues()

    def _clear_queues(self):
        """Очищает очереди"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def get_current_stats(self):
        """Возвращает текущую статистику"""
        with self._lock:
            if self.stats['total_faces'] == 0:
                return {
                    'total_faces': 0,
                    'with_helmet': 0,
                    'without': 0,
                    'compliance': 0
                }

            compliance = (self.stats['total_with_helmet'] / self.stats['total_faces']) * 100

            return {
                'total_faces': self.stats['total_faces'],
                'with_helmet': self.stats['total_with_helmet'],
                'without': self.stats['total_without'],
                'compliance': compliance
            }

    def get_progress(self):
        """Возвращает прогресс воспроизведения (0-100)"""
        if self.total_frames == 0:
            return 0
        return (self.current_frame_num / self.total_frames) * 100

    def is_finished(self):
        """Проверка, завершено ли видео"""
        return not self.is_playing and self.current_frame_num >= self.total_frames

    def close(self):
        """Закрывает видео"""
        self.is_playing = False
        self.is_paused = False

        if self.thread_read and self.thread_read.is_alive():
            self.thread_read.join(timeout=1.0)

        if self.thread_process and self.thread_process.is_alive():
            self.thread_process.join(timeout=1.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self._clear_queues()