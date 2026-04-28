import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np

import config
from detection import FaceDetector, HelmetAnalyzer
from utils import ImageProcessor
from ui.styles import UIStyles

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from utils.video_processor import VideoProcessor
    _VIDEO_PROCESSOR_AVAILABLE = True
except ImportError:
    _VIDEO_PROCESSOR_AVAILABLE = False


class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Safety Vision • Контроль касок")
        self.root.geometry("1200x750")
        self.root.configure(bg=UIStyles.BG_PRIMARY)

        # Инициализация детекторов
        self.face_detector = FaceDetector()
        self.helmet_analyzer = HelmetAnalyzer()
        self.image_processor = ImageProcessor()

        # Состояние
        self.current_image = None
        self.photoimage = None
        self.cap = None
        self.camera_thread = None
        self.camera_running = False
        self.lock = threading.Lock()

        # Для видео
        self.video_processor = None
        self.video_playing = False
        self.video_update_job = None
        self.video_controls_frame = None
        self.btn_video_play = None
        self.btn_video_stop = None
        self.video_progress = None
        self.video_time_label = None
        self.video_stats_label = None

        # Стабилизация
        self._stability_required_frames = config.STABILITY_REQUIRED_FRAMES
        self._face_presence_count = {}

        self.setup_ui()

    def setup_ui(self):
        main_container = tk.Frame(self.root, bg=UIStyles.BG_PRIMARY)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Заголовок (лаконичный)
        header_frame = tk.Frame(main_container, bg=UIStyles.BG_PRIMARY)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        title_label = tk.Label(header_frame, text="Safety Vision",
                               font=('Segoe UI', 16, 'normal'), fg=UIStyles.TEXT_PRIMARY, bg=UIStyles.BG_PRIMARY)
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(header_frame, text="Контроль защитных касок",
                                  font=('Segoe UI', 10), fg=UIStyles.TEXT_SECONDARY, bg=UIStyles.BG_PRIMARY)
        subtitle_label.pack(side=tk.LEFT, padx=(12, 0))

        # Панель кнопок
        self._create_button_panel(main_container)

        # Основной контент
        content_frame = tk.Frame(main_container, bg=UIStyles.BG_PRIMARY)
        content_frame.pack(fill=tk.BOTH, expand=True)

        self._create_image_panel(content_frame)
        self._create_results_panel(content_frame)
        self._create_status_bar(main_container)

        self.root.after(200, self._redraw_canvas_if_needed)

    def _create_button_panel(self, parent):
        control_frame = tk.Frame(parent, bg=UIStyles.BG_PRIMARY)
        control_frame.pack(fill=tk.X, pady=(0, 12))

        button_container = tk.Frame(control_frame, bg=UIStyles.BG_SECONDARY, bd=1, relief='flat')
        button_container.pack(fill=tk.X, padx=0, pady=0)

        btn_frame = tk.Frame(button_container, bg=UIStyles.BG_SECONDARY)
        btn_frame.pack(pady=8, padx=12)

        # Все кнопки в едином стиле
        self.btn_load = tk.Button(btn_frame, text="Загрузить", command=self.load_image)
        self.btn_video = tk.Button(btn_frame, text="Загрузить видео", command=self.load_video)
        self.btn_camera = tk.Button(btn_frame, text="Камера", command=self.toggle_camera)
        self.btn_analyze = tk.Button(btn_frame, text="Анализ", command=self.analyze_current)
        self.btn_clear = tk.Button(btn_frame, text="Очистить", command=self.clear_all)

        # Применяем единый стиль (без разноцветных кнопок)
        for btn in [self.btn_load, self.btn_video, self.btn_camera, self.btn_analyze, self.btn_clear]:
            UIStyles.apply_button_style(btn, is_primary=(btn == self.btn_analyze))

        for btn in [self.btn_load, self.btn_video, self.btn_camera, self.btn_analyze, self.btn_clear]:
            btn.pack(side=tk.LEFT, padx=4)

    def _create_image_panel(self, parent):
        left_frame = tk.Frame(parent, bg=UIStyles.BG_PRIMARY)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Лаконичная рамка
        canvas_container = tk.Frame(left_frame, bg=UIStyles.BORDER, bd=0)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg=UIStyles.BG_TERTIARY, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self.canvas.create_text(400, 250, text="Загрузите изображение или включите камеру",
                                font=('Segoe UI', 12), fill=UIStyles.TEXT_DISABLED, justify=tk.CENTER, tags=("hint",))

        # Привязываем canvas_frame для видео-контролов
        self.canvas_frame = canvas_container

    def _create_results_panel(self, parent):
        right_frame = tk.Frame(parent, bg=UIStyles.BG_PRIMARY, width=320)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_frame.pack_propagate(False)
        right_frame.config(width=320)

        # Статистика - компактные карточки
        stats_container = tk.Frame(right_frame, bg=UIStyles.BG_PRIMARY)
        stats_container.pack(fill=tk.X, pady=(0, 12))

        self.stats_vars = {
            'total': tk.StringVar(value="0"),
            'with_helmet': tk.StringVar(value="0"),
            'without': tk.StringVar(value="0"),
            'compliance': tk.StringVar(value="0%")
        }

        stats_grid = tk.Frame(stats_container, bg=UIStyles.BG_PRIMARY)
        stats_grid.pack(fill=tk.X)

        # Карточки статистики (однотипные)
        stats_items = [
            ('Всего', 'total'),
            ('В касках', 'with_helmet'),
            ('Без касок', 'without'),
            ('Соблюдение', 'compliance')
        ]

        for i, (label, key) in enumerate(stats_items):
            card = tk.Frame(stats_grid, bg=UIStyles.BG_SECONDARY, bd=1, relief='flat')
            card.grid(row=i // 2, column=i % 2, padx=4, pady=4, sticky='nsew')
            card.configure(highlightbackground=UIStyles.BORDER, highlightthickness=1)

            value_label = tk.Label(card, textvariable=self.stats_vars[key],
                                   font=('Segoe UI', 22, 'normal'), fg=UIStyles.TEXT_PRIMARY, bg=UIStyles.BG_SECONDARY)
            value_label.pack(pady=(10, 2))

            desc_label = tk.Label(card, text=label, font=('Segoe UI', 9),
                                  fg=UIStyles.TEXT_SECONDARY, bg=UIStyles.BG_SECONDARY)
            desc_label.pack(pady=(0, 10))

            stats_grid.grid_columnconfigure(i % 2, weight=1)
            stats_grid.grid_rowconfigure(i // 2, weight=1)

        # Разделитель (простая линия)
        separator = tk.Frame(right_frame, bg=UIStyles.BORDER, height=1)
        separator.pack(fill=tk.X, pady=8)

        # Детальный отчет - минималистичный
        details_label = tk.Label(right_frame, text="Отчет",
                                 font=('Segoe UI', 10, 'normal'), bg=UIStyles.BG_PRIMARY, fg=UIStyles.TEXT_SECONDARY)
        details_label.pack(anchor=tk.W, pady=(0, 6))

        text_container = tk.Frame(right_frame, bg=UIStyles.BORDER, bd=0)
        text_container.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(text_container, wrap=tk.WORD, font=('Consolas', 9),
                                    bg=UIStyles.BG_TERTIARY, fg=UIStyles.TEXT_PRIMARY,
                                    relief='flat', bd=0, padx=8, pady=6,
                                    selectbackground=UIStyles.ACCENT_BLUE)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, pady=1)

        scrollbar = ttk.Scrollbar(text_container, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Настройка стиля скроллбара под темную тему
        style = ttk.Style()
        style.configure("Vertical.TScrollbar", background=UIStyles.BG_SECONDARY, troughcolor=UIStyles.BG_PRIMARY,
                        arrowcolor=UIStyles.TEXT_PRIMARY)
        scrollbar.configure(style="Vertical.TScrollbar")

    def _create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg=UIStyles.BG_PRIMARY)
        status_frame.pack(fill=tk.X, pady=(12, 0))

        status_bar = tk.Frame(status_frame, bg=UIStyles.BG_SECONDARY, bd=1, relief='flat')
        status_bar.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Готов")
        status_label = tk.Label(status_bar, textvariable=self.status_var,
                                font=('Segoe UI', 9), fg=UIStyles.TEXT_SECONDARY, bg=UIStyles.BG_SECONDARY,
                                padx=12, pady=6)
        status_label.pack(side=tk.LEFT)

    def _display_results(self, results):
        self.results_text.delete(1.0, tk.END)

        if not results:
            self.results_text.insert(tk.END, "Лица не обнаружены")
            for key in self.stats_vars:
                self.stats_vars[key].set("0")
            return

        total = len(results)
        with_helmet = sum(1 for r in results if r["has_helmet"])
        without = total - with_helmet
        compliance = (with_helmet / total) * 100 if total else 0.0

        self.stats_vars['total'].set(str(total))
        self.stats_vars['with_helmet'].set(str(with_helmet))
        self.stats_vars['without'].set(str(without))
        self.stats_vars['compliance'].set(f"{compliance:.1f}%")

        # Лаконичный отчет без лишних символов
        self.results_text.insert(tk.END, f"Обнаружено: {total} лиц\n\n")

        for i, r in enumerate(results, 1):
            status = "В каске" if r['has_helmet'] else "Без каски"
            confidence = f"{int(r['score'] * 100)}%"
            self.results_text.insert(tk.END, f"{i}. {status}  [{confidence}]\n")

        self.results_text.insert(tk.END, f"\nСоблюдение: {compliance:.1f}%")
        self.results_text.see(tk.END)

    def _update_video_results(self, results, frame_num):
        """Обновляет отображение результатов для видео (лаконично)"""
        if not results:
            return

        self.results_text.delete(1.0, tk.END)

        with_helmet = sum(1 for r in results if r['has_helmet'])

        self.results_text.insert(tk.END, f"Кадр {frame_num}\n")
        self.results_text.insert(tk.END, f"В касках: {with_helmet} / {len(results)}\n\n")

        for i, r in enumerate(results, 1):
            status = "✓" if r['has_helmet'] else "✗"
            self.results_text.insert(tk.END, f"{i}. {status}  [{int(r['score'] * 100)}%]\n")

        self.results_text.see(tk.END)

    def _create_video_controls(self):
        """Создает панель управления видео (лаконичная)"""
        if hasattr(self, 'video_controls_frame') and self.video_controls_frame:
            try:
                self.video_controls_frame.destroy()
            except:
                pass

        self.video_controls_frame = tk.Frame(self.canvas_frame, bg=UIStyles.BG_SECONDARY)
        self.video_controls_frame.pack(side=tk.BOTTOM, fill=tk.X)

        btn_frame = tk.Frame(self.video_controls_frame, bg=UIStyles.BG_SECONDARY)
        btn_frame.pack(pady=6)

        self.btn_video_play = tk.Button(btn_frame, text="⏸", command=self.toggle_video_pause,
                                        font=('Segoe UI', 10), width=3, relief='flat', cursor='hand2')
        self.btn_video_stop = tk.Button(btn_frame, text="⏹", command=self.stop_video,
                                        font=('Segoe UI', 10), width=3, relief='flat', cursor='hand2')

        for btn in [self.btn_video_play, self.btn_video_stop]:
            btn.configure(bg=UIStyles.BG_TERTIARY, fg=UIStyles.TEXT_PRIMARY,
                          activebackground=UIStyles.HOVER, activeforeground=UIStyles.TEXT_PRIMARY,
                          bd=0)
            btn.pack(side=tk.LEFT, padx=2)

        progress_frame = tk.Frame(self.video_controls_frame, bg=UIStyles.BG_SECONDARY)
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.video_progress = ttk.Scale(progress_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                        command=self.seek_video)
        self.video_progress.pack(fill=tk.X)

        self.video_time_label = tk.Label(progress_frame, text="00:00 / 00:00",
                                         font=('Segoe UI', 8), bg=UIStyles.BG_SECONDARY, fg=UIStyles.TEXT_DISABLED)
        self.video_time_label.pack(pady=(4, 0))

    def display_image(self, pil_img):
        if pil_img is None:
            return
        cw = max(self.canvas.winfo_width(), 320)
        ch = max(self.canvas.winfo_height(), 240)
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih) * 0.95
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img_resized = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photoimage = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photoimage, anchor=tk.CENTER)

    def _detect_helmets(self, pil_image):
        img_rgb = np.array(pil_image.convert("RGB"))
        h_img, w_img = img_rgb.shape[:2]

        img_rgb = self.image_processor.enhance_contrast(img_rgb)
        faces = self.face_detector.detect_faces(img_rgb)

        if not faces:
            return pil_image, []

        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)

        try:
            font = ImageFont.truetype(config.FONT_PATH, size=12)
        except Exception:
            font = ImageFont.load_default()

        results = []
        new_presence = {}

        for idx, (x, y, w, h) in enumerate(faces):
            x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
            if x + w > w_img:
                w = w_img - x
            if y + h > h_img:
                h = h_img - y

            helmet_x1, helmet_y1, helmet_x2, helmet_y2 = self.image_processor.get_helmet_roi(x, y, w, h, w_img, h_img)
            helmet_roi = img_rgb[helmet_y1:helmet_y2, helmet_x1:helmet_x2]

            has_helmet, score = self.helmet_analyzer.analyze_helmet(helmet_roi)

            box_color = UIStyles.ACCENT_GREEN if has_helmet else UIStyles.ACCENT_RED
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=2)
            draw.rectangle([helmet_x1, helmet_y1, helmet_x2, helmet_y2], outline=UIStyles.ACCENT_YELLOW, width=1)

            # Лаконичная метка
            label = f"{'✓' if has_helmet else '✗'} {int(score * 100)}%"
            # Отрисовка с заливкой
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx, ty = x, max(0, helmet_y1 - text_h - 4)
            draw.rectangle([tx, ty, tx + text_w + 6, ty + text_h + 2], fill=box_color)
            draw.text((tx + 3, ty + 1), label, fill=(0, 0, 0), font=font)

            norm_key = self.image_processor.get_stability_key(x, y, w, h)
            new_presence[norm_key] = has_helmet

            results.append({
                "position": (x, y, w, h),
                "has_helmet": bool(has_helmet),
                "score": float(score)
            })

        for k, has in new_presence.items():
            prev = self._face_presence_count.get(k, 0)
            self._face_presence_count[k] = prev + 1 if has else max(0, prev - 1)

        filtered_results = []
        for r in results:
            x, y, w, h = r["position"]
            key = self.image_processor.get_stability_key(x, y, w, h)
            stable_count = self._face_presence_count.get(key, 0)
            if stable_count < self._stability_required_frames:
                r["score"] = r["score"] * (stable_count / self._stability_required_frames)
                if r["score"] < 0.15:
                    r["has_helmet"] = False
            filtered_results.append(r)
        filtered_results = self._filter_results_by_confidence(filtered_results, min_confidence=0.3)
        return annotated, filtered_results

    def _redraw_canvas_if_needed(self):
        if self.current_image is not None:
            self.display_image(self.current_image)

    def _filter_results_by_confidence(self, results, min_confidence=0.3):
        """Отсеивает результаты с низкой уверенностью"""
        return [r for r in results if r['score'] >= min_confidence]

    # ------------------- Работа с изображениями -------------------
    def load_image(self):
        filetypes = [("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=filetypes)
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.stop_camera()
            self.stop_video()
            self.current_image = img
            self.display_image(img)
            self.status_var.set(f"Загружено: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")

    # ------------------- Работа с видео -------------------
    def load_video(self):
        """Загружает видеофайл"""
        if not _VIDEO_PROCESSOR_AVAILABLE:
            messagebox.showwarning("Видео недоступно", "Модуль видео процессора не найден")
            return

        filetypes = [("Видео", "*.mp4 *.avi *.mov *.mkv *.flv"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(title="Выберите видеофайл", filetypes=filetypes)
        if not path:
            return

        try:
            # Останавливаем камеру и текущее видео
            self.stop_camera()
            self.stop_video()

            # Очищаем результаты
            for key in self.stats_vars:
                self.stats_vars[key].set("0")
            self.results_text.delete(1.0, tk.END)

            # Создаем процессор видео
            self.video_processor = VideoProcessor(process_every_n_frames=2)
            video_info = self.video_processor.open_video(path)

            # Показываем информацию о видео
            duration = video_info['duration']
            minutes = int(duration // 60)
            seconds = int(duration % 60)

            self.status_var.set(
                f"Загружено видео: {os.path.basename(path)} ({minutes}:{seconds:02d}, {video_info['fps']:.1f} FPS)")

            # Показываем первый кадр
            ret, first_frame = self.video_processor.cap.read()
            if ret:
                first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                self.current_image = Image.fromarray(first_frame_rgb)
                self.display_image(self.current_image)

            # Добавляем элементы управления видео
            self._create_video_controls()

            # Запускаем воспроизведение
            self.start_video_playback()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить видео: {e}")

    def start_video_playback(self):
        """Запускает воспроизведение видео с анализом"""
        if not self.video_processor:
            return

        self.video_playing = True

        # Запускаем воспроизведение с callback для обработки
        self.video_processor.start_playback(process_callback=self._process_video_frame)

        # Запускаем обновление UI
        self._update_video_display()

    def _process_video_frame(self, image_rgb, frame_num):
        """Обрабатывает один кадр видео"""
        # Анализируем кадр
        annotated, results = self._detect_helmets_on_image(image_rgb)

        # Форматируем результаты для статистики
        formatted_results = []
        for r in results:
            formatted_results.append({
                'has_helmet': r.get('has_helmet', False),
                'score': r.get('score', 0),
                'position': r.get('position', (0, 0, 0, 0))
            })

        # Обновляем статистику на основе результатов
        with_helmet = sum(1 for r in formatted_results if r['has_helmet'])
        without = len(formatted_results) - with_helmet

        # Обновляем UI статистику (будет в основном потоке)
        self.root.after(0, lambda: self._update_video_stats(len(formatted_results), with_helmet, without, frame_num))

        return formatted_results

    def _update_video_stats(self, total, with_helmet, without, frame_num):
        """Обновляет статистику видео в UI"""
        # Накопляем статистику
        current_total = int(self.stats_vars['total'].get())
        current_with = int(self.stats_vars['with_helmet'].get())

        self.stats_vars['total'].set(str(current_total + total))
        self.stats_vars['with_helmet'].set(str(current_with + with_helmet))
        self.stats_vars['without'].set(str(current_total + total - current_with - with_helmet))

        new_total = current_total + total
        if new_total > 0:
            compliance = ((current_with + with_helmet) / new_total) * 100
            self.stats_vars['compliance'].set(f"{compliance:.1f}%")

    def _detect_helmets_on_image(self, image_rgb):
        """Анализирует изображение (numpy array) на наличие лиц и касок"""
        h_img, w_img = image_rgb.shape[:2]

        # Улучшение контраста
        image_rgb = self.image_processor.enhance_contrast(image_rgb)

        # Детекция лиц
        faces = self.face_detector.detect_faces(image_rgb)

        if not faces:
            return Image.fromarray(image_rgb), []

        # Аннотируем изображение
        annotated_pil = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(annotated_pil)

        try:
            font = ImageFont.truetype(config.FONT_PATH, size=14)
        except Exception:
            font = ImageFont.load_default()

        results = []

        for idx, (x, y, w, h) in enumerate(faces):
            x, y, w, h = max(0, int(x)), max(0, int(y)), int(w), int(h)
            if x + w > w_img:
                w = w_img - x
            if y + h > h_img:
                h = h_img - y

            # ROI каски
            helmet_x1, helmet_y1, helmet_x2, helmet_y2 = self.image_processor.get_helmet_roi(x, y, w, h, w_img, h_img)
            helmet_roi = image_rgb[helmet_y1:helmet_y2, helmet_x1:helmet_x2]

            # Анализ
            has_helmet, score = self.helmet_analyzer.analyze_helmet(helmet_roi)

            # Отрисовка
            box_color = (46, 204, 113) if has_helmet else (231, 76, 60)
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=3)
            draw.rectangle([helmet_x1, helmet_y1, helmet_x2, helmet_y2], outline=(241, 196, 15), width=2)

            label = f"#{idx + 1} | {'В КАСКЕ' if has_helmet else 'БЕЗ КАСКИ'} | {int(score * 100)}%"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            tx, ty = x, max(0, helmet_y1 - text_h - 6)
            draw.rectangle([tx, ty, tx + text_w + 8, ty + text_h + 4], fill=box_color)
            draw.text((tx + 4, ty + 2), label, fill=(255, 255, 255), font=font)

            results.append({
                "position": (x, y, w, h),
                "has_helmet": bool(has_helmet),
                "score": float(score)
            })

        return annotated_pil, results

    def _update_video_display(self):
        """Обновляет отображение видео"""
        if not self.video_playing or not self.video_processor:
            return

        # Получаем следующий результат
        result = self.video_processor.get_next_result(timeout=0.001)

        if result:
            # Обновляем изображение
            self.current_image = Image.fromarray(result['image'])
            self.display_image(self.current_image)

            # Обновляем прогресс
            progress = self.video_processor.get_progress()
            if self.video_progress:
                self.video_progress.set(progress)

            # Обновляем время
            if hasattr(self.video_processor, 'total_frames') and self.video_processor.total_frames > 0:
                if hasattr(self.video_processor, 'fps') and self.video_processor.fps > 0:
                    current_time = self.video_processor.current_frame_num / self.video_processor.fps
                    total_time = self.video_processor.total_frames / self.video_processor.fps

                    current_str = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"
                    total_str = f"{int(total_time // 60):02d}:{int(total_time % 60):02d}"
                    if self.video_time_label:
                        self.video_time_label.config(text=f"{current_str} / {total_str}")

        # Проверяем завершение
        if self.video_processor.is_finished():
            self.stop_video()
            self.status_var.set("Видео завершено")
            messagebox.showinfo("Завершено", "Анализ видео завершен!")
            return

        # Планируем следующее обновление
        if self.video_playing:
            self.video_update_job = self.root.after(33, self._update_video_display)

    def toggle_video_pause(self):
        """Пауза/возобновление видео"""
        if not self.video_processor:
            return

        if self.video_processor.is_paused:
            self.video_processor.resume()
            if self.btn_video_play:
                self.btn_video_play.config(text="⏸")
                # Убираем установку цвета, так как он и так правильный
            self.status_var.set("Воспроизведение возобновлено")
        else:
            self.video_processor.pause()
            if self.btn_video_play:
                self.btn_video_play.config(text="▶")
            self.status_var.set("Видео на паузе")

    def seek_video(self, value):
        """Перемотка видео"""
        if not self.video_processor:
            return

        was_paused = self.video_processor.is_paused
        if not was_paused:
            self.video_processor.pause()

        self.video_processor.seek(float(value))
        self.status_var.set(f"Перемотка: {float(value):.1f}%")

        if not was_paused:
            self.root.after(100, self.video_processor.resume)

    def stop_video(self):
        """Останавливает воспроизведение видео"""
        self.video_playing = False

        if self.video_update_job:
            try:
                self.root.after_cancel(self.video_update_job)
            except:
                pass
            self.video_update_job = None

        if self.video_processor:
            self.video_processor.close()
            self.video_processor = None

        # Удаляем элементы управления
        if hasattr(self, 'video_controls_frame') and self.video_controls_frame:
            try:
                self.video_controls_frame.destroy()
            except:
                pass
            self.video_controls_frame = None

        self.btn_video_play = None
        self.btn_video_stop = None
        self.video_progress = None
        self.video_time_label = None
        self.video_stats_label = None

        self.status_var.set("Видео остановлено")

    # ------------------- Работа с камерой -------------------
    def toggle_camera(self):
        if not _CV2_AVAILABLE:
            messagebox.showwarning("Камера недоступна", "OpenCV не установлен")
            return
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if not _CV2_AVAILABLE or self.camera_running:
            return
        try:
            self.stop_video()
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
        except Exception as e:
            messagebox.showerror("Камера", f"Ошибка открытия камеры: {e}")
            return

        self.camera_running = True
        self.btn_camera.config(text="Камера (вкл)")
        self.status_var.set("Камера активна")
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _camera_loop(self):
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            with self.lock:
                self.current_image = img
            self.root.after(0, lambda im=img.copy(): self.display_image(im))
            time.sleep(config.CAMERA_FRAME_DELAY)

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.camera_running = False
        self.root.after(0, lambda: self.btn_camera.config(text="Камера"))
        self.root.after(0, lambda: self.status_var.set("Камера отключена"))

    def stop_camera(self):
        if self.camera_running:
            self.camera_running = False
        else:
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            self.camera_running = False
            self.btn_camera.config(text="Камера")
            self.status_var.set("Камера отключена")

    # ------------------- Анализ изображения -------------------
    def analyze_current(self):
        with self.lock:
            img = self.current_image.copy() if self.current_image is not None else None
        if img is None:
            messagebox.showwarning("Нет изображения", "Сначала загрузите изображение или включите камеру")
            return

        self.status_var.set("Анализ изображения...")
        self.root.update_idletasks()

        try:
            annotated, results = self._detect_helmets(img)
        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Ошибка во время анализа: {e}")
            self.status_var.set("Ошибка анализа")
            return

        self.current_image = annotated
        self.display_image(annotated)
        self._display_results(results)
        self.status_var.set("Анализ завершен")

    def clear_all(self):
        self.stop_video()
        self.stop_camera()
        with self.lock:
            self.current_image = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 250, text="Загрузите изображение\nили включите камеру",
                                font=('Segoe UI', 12), fill=UIStyles.TEXT_DISABLED, justify=tk.CENTER, tags=("hint",))
        self.results_text.delete(1.0, tk.END)
        for key in self.stats_vars:
            self.stats_vars[key].set("0")
        self.status_var.set("Готов")
