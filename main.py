# main.py
import tkinter as tk
from ui.app import HelmetDetectionApp

def main():
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    
    # Центрирование окна
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    x, y = (sw - w) // 2, (sh - h) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()