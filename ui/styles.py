# ui/styles.py
import tkinter as tk
from tkinter import ttk


class UIStyles:
    """Лаконичная цветовая схема и стили интерфейса"""

    # Основные цвета (монохромная гамма с акцентами)
    BG_PRIMARY = '#1e1e1e'  # Тёмный фон окна
    BG_SECONDARY = '#252526'  # Вторичный фон (карточки)
    BG_TERTIARY = '#2d2d30'  # Третичный фон (canvas)

    TEXT_PRIMARY = '#ffffff'  # Белый текст
    TEXT_SECONDARY = '#cccccc'  # Светло-серый текст
    TEXT_DISABLED = '#666666'  # Тёмно-серый текст

    ACCENT_GREEN = '#4ec9b0'  # Зелёный акцент (каска есть)
    ACCENT_RED = '#f14c4c'  # Красный акцент (каски нет)
    ACCENT_BLUE = '#569cd6'  # Синий акцент (кнопки, рамки)
    ACCENT_YELLOW = '#dcdcaa'  # Жёлтый акцент (область каски)

    BORDER = '#3e3e42'  # Цвет границ
    HOVER = '#3e3e42'  # Цвет при наведении

    # Цвета для обратной совместимости (для методов, которые их используют)
    PRIMARY = ACCENT_BLUE  # Синий (основной)
    SUCCESS = '#4ec9b0'  # Бирюзовый (успех)
    DANGER = '#f14c4c'  # Красный (опасность)
    WARNING = '#dcdcaa'  # Жёлтый (предупреждение)
    GRAY = TEXT_DISABLED  # Серый
    PURPLE = '#569cd6'  # Синий (вместо фиолетового)

    LIGHT_GRAY = BORDER
    WHITE = BG_SECONDARY
    DARK = TEXT_PRIMARY
    BG = BG_PRIMARY
    CANVAS_BG = BG_TERTIARY

    @classmethod
    def apply_button_style(cls, button, is_primary=False):
        """Применяет единый стиль для всех кнопок"""
        if is_primary:
            bg = cls.PRIMARY
            fg = cls.TEXT_PRIMARY
            active_bg = '#3a6a9e'
        else:
            bg = cls.BG_SECONDARY
            fg = cls.TEXT_PRIMARY
            active_bg = cls.HOVER

        button.configure(
            bg=bg,
            fg=fg,
            font=('Segoe UI', 9),
            padx=15,
            pady=6,
            relief='flat',
            cursor='hand2',
            bd=1,
            highlightthickness=0,
            activebackground=active_bg,
            activeforeground=cls.TEXT_PRIMARY
        )

    @classmethod
    def apply_stats_card_style(cls, card_frame):
        """Применяет единый стиль для карточек статистики"""
        card_frame.configure(
            bg=cls.BG_SECONDARY,
            relief='flat',
            bd=1,
            highlightbackground=cls.BORDER,
            highlightthickness=1
        )

    @classmethod
    def get_stats_card_color(cls, index):
        """Возвращает единый цвет для всех карточек"""
        return cls.BG_SECONDARY

    @classmethod
    def get_status_color(cls, has_helmet):
        """Возвращает цвет статуса"""
        return cls.ACCENT_GREEN if has_helmet else cls.ACCENT_RED