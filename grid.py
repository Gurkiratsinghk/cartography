import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class GridGenerator:
    def __init__(self, master):
        self.master = master
        master.title("Cartography Grid Generator")

        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.grid_cells = 10  # Default number of grid cells
        self.grid_shape = "Square"  # Default grid shape

        # UI Elements
        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.canvas = tk.Canvas(master, bg="white")
        self.canvas.pack(expand=True, fill="both")

        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=10)

        self.slider_label = tk.Label(self.control_frame, text="Grid Cells:")
        self.slider_label.pack(side=tk.LEFT, padx=5)

        self.grid_slider = tk.Scale(self.control_frame, from_=1, to_=100, orient=tk.HORIZONTAL,
                                     command=self.update_grid, length=300)
        self.grid_slider.set(self.grid_cells)
        self.grid_slider.pack(side=tk.LEFT, padx=5)

        self.shape_label = tk.Label(self.control_frame, text="Grid Shape:")
        self.shape_label.pack(side=tk.LEFT, padx=5)

        self.shape_options = ["Square", "Hexagon"]
        self.shape_var = tk.StringVar(master)
        self.shape_var.set(self.grid_shape)
        self.shape_dropdown = ttk.Combobox(self.control_frame, textvariable=self.shape_var,
                                           values=self.shape_options, state="readonly")
        self.shape_dropdown.pack(side=tk.LEFT, padx=5)
        self.canvas.bind("<Configure>", self.update_grid) # Bind configure event for resizing
        self.shape_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_grid())
        self.canvas.config(width=600, height=400)  # Default canvas size

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(self.image_path)
            self.display_image = None  # Clear previous display image
            self.update_grid() # Call update_grid to display the loaded image



    def update_grid(self, event=None):
        self.canvas.delete("all") # Clear previous drawings
        if self.original_image:
            self.grid_cells = self.grid_slider.get()
            self.grid_shape = self.shape_var.get()

            # Resize image to fit canvas
            canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 600
            canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 400

            img_width, img_height = self.original_image.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(resized_image) # Keep a reference!
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, image=self.display_image, anchor=tk.CENTER) # Display the image

            # Calculate image top-left corner on canvas
            img_x_offset = (canvas_width - new_width) / 2
            img_y_offset = (canvas_height - new_height) / 2

            if self.grid_shape == "Square":
                self._draw_square_grid(new_width, new_height, img_x_offset, img_y_offset)
            elif self.grid_shape == "Hexagon":
                self._draw_hexagon_grid(new_width, new_height, img_x_offset, img_y_offset)

    def _draw_square_grid(self, width, height, x_offset, y_offset):
        """Draws a square grid over the image."""
        cell_size_x = width / self.grid_cells
        cell_size_y = height / self.grid_cells
        for i in range(self.grid_cells):
            for j in range(self.grid_cells):
                x1 = x_offset + i * cell_size_x
                y1 = y_offset + j * cell_size_y
                x2 = x_offset + (i + 1) * cell_size_x
                y2 = y_offset + (j + 1) * cell_size_y
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=1)

    def _draw_hexagon_grid(self, width, height, x_offset, y_offset):
        """Draws a hexagonal grid over the image."""
        # Fit the middle row with hexagons based on the slider value
        hex_count_in_row = self.grid_cells
        hex_width = width / hex_count_in_row
        side_length = hex_width / 2
        hex_height = np.sqrt(3) * side_length
        
        # Vertical distance between centers of staggered rows
        vert_dist_between_rows = hex_height * 0.75

        # Calculate how many rows fit vertically (centered on image)
        num_rows = int(height // vert_dist_between_rows)
        if num_rows % 2 == 0:
            num_rows += 1  # Ensure odd number for a true middle row

        # Center the grid vertically on the image
        total_grid_height = (num_rows - 1) * vert_dist_between_rows + hex_height
        y_start = y_offset + (height - total_grid_height) / 2

        for row in range(num_rows):
            # Stagger odd rows horizontally
            row_offset_x = hex_width / 2 if row % 2 == 1 else 0
            # Staggered rows have one less hexagon to fit
            hex_count_this_row = hex_count_in_row if row % 2 == 0 else hex_count_in_row - 1
            x_start = x_offset + row_offset_x + side_length

            for col in range(hex_count_this_row):
                center_x = x_start + col * hex_width
                center_y = y_start + row * vert_dist_between_rows + hex_height / 2

                # Calculate 6 vertices for a pointy-top hexagon
                points = []
                for i in range(6):
                    angle_rad = np.deg2rad(60 * i - 30)
                    x = center_x + side_length * np.cos(angle_rad)
                    y = center_y + side_length * np.sin(angle_rad)
                    points.append((x, y))
                self.canvas.create_polygon(points, outline="red", width=1, fill="")

root = tk.Tk()
app = GridGenerator(root)
root.mainloop()
