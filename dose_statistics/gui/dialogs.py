# gui/dialogs.py
from tkinter import Tk, simpledialog, filedialog
import os

def ask_text(title: str, prompt: str, default: str = "") -> str:
    root = Tk(); root.withdraw()
    val = simpledialog.askstring(title, prompt, initialvalue=default)
    root.destroy()
    if val is None:
        raise KeyboardInterrupt("User cancelled input.")
    return val

def ask_float(title: str, prompt: str, default: float = 0.0) -> float:
    s = ask_text(title, prompt, str(default))
    return float(s)

def ask_yesno(title: str, prompt: str, default_yes: bool = True) -> bool:
    default = "y" if default_yes else "n"
    s = ask_text(title, f"{prompt} [y/n]", default).strip().lower()
    return s.startswith("y")

def ask_open_file(title: str = "Select image file") -> tuple[str, str]:
    root = Tk(); root.withdraw()
    fpath = filedialog.askopenfilename(title=title)
    root.destroy()
    if not fpath:
        raise KeyboardInterrupt("User cancelled file selection.")
    return os.path.basename(fpath), os.path.dirname(fpath)
