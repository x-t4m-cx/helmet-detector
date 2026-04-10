# ui/styles.py
import tkinter as tk
from tkinter import ttk


class UIStyles:
    """Цветовая схема и стили интерфейса"""
    
    # Цвета
    PRIMARY = '#3498db'      # Синий
    SUCCESS = '#2ecc71'      # Зеленый
    WARNING = '#f39c12'      # Оранжевый
    DANGER = '#e74c3c'       # Красный
    PURPLE = '#9b59b6'       # Фиолетовый
    DARK = '#2c3e50'         # Темно-синий
    GRAY = '#7f8c8d'         # Серый
    LIGHT_GRAY = '#ecf0f1'   # Светло-серый
    WHITE = '#ffffff'        # Белый
    BG = '#f0f0f0'          # Фон окна
    CANVAS_BG = '#f8f9fa'   # Фон canvas
    
    @classmethod
    def apply_button_style(cls, button, color, hover_color=None):
        """Применяет стиль к кнопке"""
        if hover_color is None:
            hover_color = color
        
        button.configure(
            bg=color,
            fg=cls.WHITE,
            font=('Segoe UI', 10),
            padx=15,
            pady=8,
            relief='flat',
            cursor='hand2',
            activebackground=hover_color,
            activeforeground=cls.WHITE
        )
    
    @classmethod
    def get_stats_card_color(cls, index):
        """Возвращает цвет для карточки статистики"""
        colors = [cls.PRIMARY, cls.SUCCESS, cls.DANGER, cls.PURPLE]
        return colors[index % len(colors)]