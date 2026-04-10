# ui/app.py
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


class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Детектор защитной каски")
        self.root.geometry("1200x750")
        self.root.configure(bg=UIStyles.BG)
        
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
        
        # Стабилизация
        self._stability_required_frames = config.STABILITY_REQUIRED_FRAMES
        self._face_presence_count = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        main_container = tk.Frame(self.root, bg=UIStyles.BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        header_frame = tk.Frame(main_container, bg=UIStyles.BG)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="Детектор защитной каски",
                               font=('Segoe UI', 18, 'bold'), fg=UIStyles.DARK, bg=UIStyles.BG)
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(header_frame, text="Анализ соблюдения правил безопасности",
                                  font=('Segoe UI', 10), fg=UIStyles.GRAY, bg=UIStyles.BG)
        subtitle_label.pack(side=tk.LEFT, padx=(15, 0))
        
        # Панель кнопок
        self._create_button_panel(main_container)
        
        # Основной контент
        content_frame = tk.Frame(main_container, bg=UIStyles.BG)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель - изображение
        self._create_image_panel(content_frame)
        
        # Правая панель - результаты
        self._create_results_panel(content_frame)
        
        # Статусная строка
        self._create_status_bar(main_container)
        
        self.root.after(200, self._redraw_canvas_if_needed)
    
    def _create_button_panel(self, parent):
        control_frame = tk.Frame(parent, bg=UIStyles.BG)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        button_card = tk.Frame(control_frame, bg=UIStyles.WHITE, relief='flat', bd=1)
        button_card.pack(fill=tk.X)
        
        button_container = tk.Frame(button_card, bg=UIStyles.WHITE)
        button_container.pack(pady=12, padx=12)
        
        self.btn_load = tk.Button(button_container, text="Загрузить изображение", command=self.load_image)
        self.btn_camera = tk.Button(button_container, text="Включить камеру", command=self.toggle_camera)
        self.btn_analyze = tk.Button(button_container, text="Анализировать", command=self.analyze_current)
        self.btn_clear = tk.Button(button_container, text="Очистить", command=self.clear_all)
        
        UIStyles.apply_button_style(self.btn_load, UIStyles.PRIMARY, '#2980b9')
        UIStyles.apply_button_style(self.btn_camera, UIStyles.SUCCESS, '#27ae60')
        UIStyles.apply_button_style(self.btn_analyze, UIStyles.PURPLE, '#8e44ad')
        UIStyles.apply_button_style(self.btn_clear, UIStyles.DANGER, '#c0392b')
        
        for btn in [self.btn_load, self.btn_camera, self.btn_analyze, self.btn_clear]:
            btn.pack(side=tk.LEFT, padx=5)
    
    def _create_image_panel(self, parent):
        left_frame = tk.Frame(parent, bg=UIStyles.BG)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        image_card = tk.LabelFrame(left_frame, text="Изображение", 
                                   font=('Segoe UI', 11, 'bold'), fg=UIStyles.DARK,
                                   bg=UIStyles.WHITE, bd=1, relief='solid')
        image_card.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_frame = tk.Frame(image_card, bg=UIStyles.WHITE)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg=UIStyles.CANVAS_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.canvas.create_text(400, 250, text="Загрузите изображение\nили включите камеру",
                                font=('Segoe UI', 14), fill=UIStyles.GRAY, justify=tk.CENTER, tags=("hint",))
    
    def _create_results_panel(self, parent):
        # Исправлено: задаем ширину через config, а не через pack
        right_frame = tk.Frame(parent, bg=UIStyles.BG, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right_frame.pack_propagate(False)  # Запрещаем изменять размер
        right_frame.config(width=350)
        
        results_card = tk.LabelFrame(right_frame, text="Результаты анализа",
                                     font=('Segoe UI', 11, 'bold'), fg=UIStyles.DARK,
                                     bg=UIStyles.WHITE, bd=1, relief='solid')
        results_card.pack(fill=tk.BOTH, expand=True)
        
        # Статистика
        stats_frame = tk.Frame(results_card, bg=UIStyles.WHITE)
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.stats_vars = {
            'total': tk.StringVar(value="0"),
            'with_helmet': tk.StringVar(value="0"),
            'without': tk.StringVar(value="0"),
            'compliance': tk.StringVar(value="0%")
        }
        
        stats_grid = tk.Frame(stats_frame, bg=UIStyles.WHITE)
        stats_grid.pack()
        
        stats_data = [
            ('Всего сотрудников', 'total'),
            ('В касках', 'with_helmet'),
            ('Без касок', 'without'),
            ('Соблюдение', 'compliance')
        ]
        
        for i, (label, key) in enumerate(stats_data):
            color = UIStyles.get_stats_card_color(i)
            card = tk.Frame(stats_grid, bg=color, relief='flat', bd=0)
            card.grid(row=i//2, column=i%2, padx=5, pady=5, sticky='nsew')
            
            value_label = tk.Label(card, textvariable=self.stats_vars[key],
                                   font=('Segoe UI', 18, 'bold'), fg=UIStyles.WHITE, bg=color)
            value_label.pack(pady=(10, 5))
            
            desc_label = tk.Label(card, text=label, font=('Segoe UI', 10),
                                  fg=UIStyles.WHITE, bg=color)
            desc_label.pack(pady=(0, 10))
            
            stats_grid.grid_columnconfigure(i%2, weight=1)
        
        # Детальный отчет
        separator = tk.Frame(results_card, bg=UIStyles.LIGHT_GRAY, height=2)
        separator.pack(fill=tk.X, padx=10, pady=10)
        
        details_label = tk.Label(results_card, text="Детальный отчет",
                                 font=('Segoe UI', 11, 'bold'), bg=UIStyles.WHITE, fg=UIStyles.DARK)
        details_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        text_frame = tk.Frame(results_card, bg=UIStyles.WHITE)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 10),
                                    bg=UIStyles.CANVAS_BG, fg=UIStyles.DARK, relief='flat', bd=0)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def _create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg=UIStyles.BG)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        status_card = tk.Frame(status_frame, bg=UIStyles.WHITE, relief='flat', bd=1)
        status_card.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="Готов к работе")
        status_label = tk.Label(status_card, textvariable=self.status_var,
                                font=('Segoe UI', 9), fg=UIStyles.GRAY, bg=UIStyles.WHITE, padx=10, pady=8)
        status_label.pack(side=tk.LEFT)
    
    def _redraw_canvas_if_needed(self):
        if self.current_image is not None:
            self.display_image(self.current_image)
    
    # ------------------- Работа с изображениями и камерой -------------------
    def load_image(self):
        filetypes = [("Изображения", "*.jpg *.jpeg *.png *.bmp"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(title="Выберите изображение", filetypes=filetypes)
        if not path:
            return
        try:
            img = Image.open(path).convert("RGB")
            self.stop_camera()
            self.current_image = img
            self.display_image(img)
            self.status_var.set(f"Загружено: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
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
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == 'nt' else cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Не удалось открыть камеру")
        except Exception as e:
            messagebox.showerror("Камера", f"Ошибка открытия камеры: {e}")
            return
        
        self.camera_running = True
        self.btn_camera.config(text="Выключить камеру", bg=UIStyles.DANGER, activebackground='#c0392b')
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
        self.root.after(0, lambda: self.btn_camera.config(text="Включить камеру", bg=UIStyles.SUCCESS, activebackground='#27ae60'))
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
            self.btn_camera.config(text="Включить камеру", bg=UIStyles.SUCCESS, activebackground='#27ae60')
            self.status_var.set("Камера отключена")
    
    def clear_all(self):
        self.stop_camera()
        with self.lock:
            self.current_image = None
        self.canvas.delete("all")
        self.canvas.create_text(400, 250, text="Загрузите изображение\nили включите камеру",
                                font=('Segoe UI', 14), fill=UIStyles.GRAY, justify=tk.CENTER, tags=("hint",))
        self.results_text.delete(1.0, tk.END)
        for key in self.stats_vars:
            self.stats_vars[key].set("0")
        self.status_var.set("Готов к работе")
    
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
    
    def _detect_helmets(self, pil_image):
        img_rgb = np.array(pil_image.convert("RGB"))
        h_img, w_img = img_rgb.shape[:2]
        
        # Улучшение контраста
        img_rgb = self.image_processor.enhance_contrast(img_rgb)
        
        # Детекция лиц
        faces = self.face_detector.detect_faces(img_rgb)
        
        if not faces:
            return pil_image, []
        
        # Аннотация
        annotated = pil_image.copy()
        draw = ImageDraw.Draw(annotated)
        
        try:
            font = ImageFont.truetype(config.FONT_PATH, size=14)
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
            
            # ROI каски
            helmet_x1, helmet_y1, helmet_x2, helmet_y2 = self.image_processor.get_helmet_roi(x, y, w, h, w_img, h_img)
            helmet_roi = img_rgb[helmet_y1:helmet_y2, helmet_x1:helmet_x2]
            
            # Анализ
            has_helmet, score = self.helmet_analyzer.analyze_helmet(helmet_roi)
            
            # Ключ для стабилизации
            norm_key = self.image_processor.get_stability_key(x, y, w, h)
            new_presence[norm_key] = has_helmet
            
            # Отрисовка
            box_color = (46, 204, 113) if has_helmet else (231, 76, 60)
            draw.rectangle([x, y, x + w, y + h], outline=box_color, width=3)
            draw.rectangle([helmet_x1, helmet_y1, helmet_x2, helmet_y2], outline=(241, 196, 15), width=2)
            
            label = f"#{idx+1} | {'В КАСКЕ' if has_helmet else 'БЕЗ КАСКИ'} | {int(score*100)}%"
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
        
        # Стабилизация
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
        
        return annotated, filtered_results
    
    def _display_results(self, results):
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "Области интереса не обнаружены.\nПопробуйте другое изображение.")
            for key in self.stats_vars:
                self.stats_vars[key].set("0")
            return
        
        total = len(results)
        with_helmet = sum(1 for r in results if r["has_helmet"])
        without = total - with_helmet
        compliance = (with_helmet / total) * 100 if total else 0.0
        
        # Обновление статистики
        self.stats_vars['total'].set(str(total))
        self.stats_vars['with_helmet'].set(str(with_helmet))
        self.stats_vars['without'].set(str(without))
        self.stats_vars['compliance'].set(f"{compliance:.1f}%")
        
        # Детальный отчет
        self.results_text.insert(tk.END, "ДЕТАЛЬНЫЙ ОТЧЕТ ПО КАЖДОМУ СОТРУДНИКУ\n")
        
        for i, r in enumerate(results, 1):
            status = "В КАСКЕ" if r['has_helmet'] else "БЕЗ КАСКИ"
            self.results_text.insert(tk.END, f"Сотрудник #{i}:\n")
            self.results_text.insert(tk.END, f"  Статус: {status}\n")
            self.results_text.insert(tk.END, f"  Уверенность: {int(r['score']*100)}%\n")
            self.results_text.insert(tk.END, f"  Позиция: x={r['position'][0]}, y={r['position'][1]}\n\n")
        
        self.results_text.insert(tk.END, f"ИТОГО: {with_helmet} из {total} сотрудников в касках\n")
        self.results_text.insert(tk.END, f"СОБЛЮДЕНИЕ: {compliance:.1f}%\n")
        
        self.results_text.see(tk.END)