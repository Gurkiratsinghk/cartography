import tkinter as tk
from src.grid import GridGenerator

if __name__ == "__main__":
    root = tk.Tk()
    app = GridGenerator(root)
    root.mainloop()
