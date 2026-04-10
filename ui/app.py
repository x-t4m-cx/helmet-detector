# ui/app.py
# Реализация подключения веб-камеры, ручного выбора источника и обработки ошибок

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Импорт настроек (если используете config.py)
try:
    import config
except ImportError:
    # Если config.py отсутствует, задаём значения по умолчанию
    config = type('', (), {})()
    config.MAX_CAMERAS_TO_CHECK = 5
    config.CAMERA_FRAME_DELAY = 0.03
    config.USE_DSHOW_ON_WINDOWS = True

# Проверка наличия OpenCV
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


def get_available_cameras(max_test=config.MAX_CAMERAS_TO_CHECK):
    """
    Определяет индексы доступных веб-камер.
    Возвращает список целых чисел (индексов), которые можно открыть.
    """
    available = []
    for i in range(max_test):
        try:
            # Пытаемся открыть камеру с разными параметрами (для совместимости)
            if os.name == 'nt' and config.USE_DSHOW_ON_WINDOWS:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                available.append(i)
                cap.release()
            else:
                # Пробуем без DSHOW, если не получилось (для некоторых камер)
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
        except Exception:
            # Игнорируем ошибки при проверке
            pass
    return available


class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор защитной каски — модуль камеры")
        self.root.geometry("900x650")

        # Состояние камеры
        self.cap = None             # объект VideoCapture
        self.camera_running = False
        self.camera_thread = None
        self.lock = threading.Lock()
        self.current_image = None   # последний кадр (PIL Image)
        self.photoimage = None      # для отображения на Canvas

        # Выбранный индекс камеры
        self.selected_camera_index = None
        self.available_cameras = []

        # Построение интерфейса
        self.setup_ui()

        # После отрисовки окна – обновить список доступных камер
        self.root.after(100, self.update_camera_list)

    def setup_ui(self):
        """Создаёт все элементы управления."""
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        header = ttk.Label(main, text="📷 Подключение веб-камеры",
                           font=("Arial", 16, "bold"))
        header.pack(anchor=tk.CENTER, pady=(0, 8))

        # Панель управления
        controls = ttk.Frame(main)
        controls.pack(fill=tk.X, pady=4)

        # Кнопка включения/выключения камеры
        self.btn_camera = ttk.Button(controls, text="▶ Включить камеру",
                                     command=self.toggle_camera)
        self.btn_camera.pack(side=tk.LEFT, padx=4)

        # Выбор источника (комбобокс)
        ttk.Label(controls, text="Выбор камеры:").pack(side=tk.LEFT, padx=(10, 4))
        self.camera_combo = ttk.Combobox(controls, state="readonly", width=15)
        self.camera_combo.pack(side=tk.LEFT, padx=4)
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        # Кнопка обновления списка камер
        btn_refresh = ttk.Button(controls, text="🔄 Обновить", command=self.update_camera_list)
        btn_refresh.pack(side=tk.LEFT, padx=4)

        # Кнопка очистки (остановка и сброс)
        btn_clear = ttk.Button(controls, text="🗑 Очистить", command=self.clear_all)
        btn_clear.pack(side=tk.LEFT, padx=4)

        # Область для отображения видео
        video_frame = ttk.LabelFrame(main, text="Видеопоток", padding=6)
        video_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.canvas = tk.Canvas(video_frame, bg="#efefef", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Подсказка, если видео нет
        self.canvas.create_text(400, 200, text="Нажмите 'Включить камеру'",
                                font=("Arial", 14), fill="gray", tags=("hint",))

        # Статусная строка
        self.status_var = tk.StringVar(value="Готово")
        status_bar = ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, padding=4)
        status_bar.pack(fill=tk.X, pady=(6, 0))

    # ------------------- Работа со списком камер -------------------
    def update_camera_list(self):
        """Обновляет выпадающий список доступными камерами."""
        self.available_cameras = get_available_cameras()
        if self.available_cameras:
            # Формируем текстовые метки для комбобокса
            labels = [f"Камера {idx}" for idx in self.available_cameras]
            self.camera_combo['values'] = labels
            self.camera_combo.current(0)
            self.selected_camera_index = self.available_cameras[0]
            self.status_var.set(f"Найдено камер: {len(self.available_cameras)}")
        else:
            self.camera_combo['values'] = []
            self.camera_combo.set("Камеры не найдены")
            self.selected_camera_index = None
            self.status_var.set("Камеры не обнаружены. Проверьте подключение.")
            # Если камера была запущена – останавливаем
            if self.camera_running:
                self.stop_camera()

    def on_camera_selected(self, event=None):
        """Вызывается при выборе другого источника из списка."""
        selection = self.camera_combo.get()
        if selection and selection.startswith("Камера "):
            try:
                idx = int(selection.split()[1])
                self.selected_camera_index = idx
                # Если камера уже запущена – перезапускаем с новым индексом
                if self.camera_running:
                    self.stop_camera()
                    self.start_camera(self.selected_camera_index)
            except (ValueError, IndexError):
                pass

    # ------------------- Управление камерой -------------------
    def toggle_camera(self):
        """Включает или выключает камеру с проверкой ошибок."""
        if not _CV2_AVAILABLE:
            messagebox.showerror(
                "Ошибка",
                "OpenCV не установлен. Установите opencv-python для работы с камерой.\n"
                "Команда: pip install opencv-python"
            )
            return

        if self.camera_running:
            self.stop_camera()
        else:
            if self.selected_camera_index is None:
                messagebox.showwarning("Нет камеры",
                                       "Нет доступных камер. Нажмите 'Обновить'.")
                return
            self.start_camera(self.selected_camera_index)

    def start_camera(self, camera_index):
        """
        Запускает захват видео с указанной камеры.
        Обрабатывает ошибки открытия и инициализации.
        """
        if not _CV2_AVAILABLE:
            return

        try:
            # Открываем камеру с учётом ОС
            if os.name == 'nt' and config.USE_DSHOW_ON_WINDOWS:
                self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(camera_index)

            # Проверка, что камера открылась
            if not self.cap.isOpened():
                # Повторная попытка без DSHOW (если была ошибка)
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    raise RuntimeError(f"Не удалось открыть камеру с индексом {camera_index}")

            # Необязательно: установка параметров (разрешение, FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        except Exception as e:
            messagebox.showerror("Ошибка камеры", f"Не удалось инициализировать камеру:\n{e}")
            self.cap = None
            return

        self.camera_running = True
        self.btn_camera.config(text="⏹ Выключить камеру")
        self.status_var.set(f"Камера {camera_index} работает")

        # Запускаем поток захвата кадров
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _camera_loop(self):
        """
        Цикл захвата кадров в отдельном потоке.
        Обрабатывает потерю кадров и автоматически останавливается при ошибке.
        """
        consecutive_errors = 0
        while self.camera_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        # Слишком много ошибок – считаем камеру недоступной
                        self.root.after(0, self._camera_fatal_error)
                        break
                    time.sleep(0.1)
                    continue
                consecutive_errors = 0

                # Конвертируем BGR → RGB → PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                with self.lock:
                    self.current_image = img

                # Обновляем изображение в главном потоке
                self.root.after(0, lambda im=img.copy(): self.display_image(im))

                time.sleep(config.CAMERA_FRAME_DELAY)  # контроль FPS

            except Exception as e:
                # Логируем ошибку, но продолжаем работу
                print(f"Ошибка в цикле камеры: {e}")
                time.sleep(0.1)

        # Завершение работы камеры
        self._release_camera()

    def _camera_fatal_error(self):
        """Вызывается при критической ошибке камеры (нет сигнала)."""
        self.stop_camera()
        messagebox.showerror("Камера", "Потерян сигнал с камеры. Проверьте подключение.")
        self.status_var.set("Ошибка камеры")

    def _release_camera(self):
        """Освобождает ресурсы камеры и обновляет UI."""
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.camera_running = False
        self.root.after(0, lambda: self.btn_camera.config(text="▶ Включить камеру"))
        self.root.after(0, lambda: self.status_var.set("Камера отключена"))

    def stop_camera(self):
        """Останавливает захват и поток камеры."""
        self.camera_running = False
        # Ждём завершения потока (не более 1 секунды)
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
        self._release_camera()

    # ------------------- Отображение и очистка -------------------
    def display_image(self, pil_img):
        """Отображает PIL изображение на Canvas с сохранением пропорций."""
        if pil_img is None:
            return
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih) * 0.95
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img_resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photoimage, anchor=tk.CENTER)

    def clear_all(self):
        """Останавливает камеру и очищает экран."""
        self.stop_camera()
        with self.lock:
            self.current_image = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 200, text="Нажмите 'Включить камеру'",
                                font=("Arial", 14), fill="gray", tags=("hint",))
        self.status_var.set("Очищено")

    # ------------------- Завершение работы -------------------
    def on_closing(self):
        """Корректное завершение приложения."""
        self.stop_camera()
        self.root.destroy()