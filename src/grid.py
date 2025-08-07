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
        self.binarized_image = None  # To store the binarized image
        self.textured_image = None # To store the textured image
        self.display_image = None
        self.grid_cells = 10  # Default number of grid cells
        self.grid_shape = "Square"  # Default grid shape

        # --- Selection Rectangle ---
        self.selection_rect = None
        self.selection_start_x = 0
        self.selection_start_y = 0

        # --- UI Elements ---

        # Top frame for buttons
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(pady=10)

        self.load_button = tk.Button(self.top_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.binarize_button = tk.Button(self.top_frame, text="Binarize Image", command=self.binarize_image)
        self.binarize_button.pack(side=tk.LEFT, padx=5)

        self.invert_var = tk.BooleanVar()
        self.invert_checkbox = tk.Checkbutton(self.top_frame, text="Invert Binarization", var=self.invert_var, command=self.binarize_image)
        self.invert_checkbox.pack(side=tk.LEFT, padx=5)

        self.texture_button = tk.Button(self.top_frame, text="Generate Textures", command=self.generate_textures)
        self.texture_button.pack(side=tk.LEFT, padx=5)

        self.rules_button = tk.Button(self.top_frame, text="Apply Rules", command=self.apply_rules)
        self.rules_button.pack(side=tk.LEFT, padx=5)

        # Canvas for displaying the image and grid
        self.canvas = tk.Canvas(master, bg="white")
        self.canvas.pack(expand=True, fill="both")
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Frame for grid controls
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=10)

        # Slider for controlling the number of grid cells
        self.slider_label = tk.Label(self.control_frame, text="Grid Cells:")
        self.slider_label.pack(side=tk.LEFT, padx=5)

        self.grid_slider = tk.Scale(self.control_frame, from_=1, to_=100, orient=tk.HORIZONTAL,
                                     command=self.update_grid, length=300)
        self.grid_slider.set(self.grid_cells)
        self.grid_slider.pack(side=tk.LEFT, padx=5)

        self.shape_label = tk.Label(self.control_frame, text="Grid Shape:")
        self.shape_label.pack(side=tk.LEFT, padx=5)

        # Dropdown for selecting the grid shape
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
            self.binarized_image = None # Reset binarized image on new load
            self.display_image = None  # Clear previous display image
            self.update_grid() # Call update_grid to display the loaded image

    def binarize_image(self):
        """Converts the original image to a black and white image."""
        if self.original_image:
            # Convert PIL image to OpenCV format
            cv_image = np.array(self.original_image.convert('RGB'))
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

            # Apply adaptive thresholding. The parameters are:
            # - gray_image: The input image.
            # - 255: The maximum value to use.
            # - cv2.ADAPTIVE_THRESH_GAUSSIAN_C: The adaptive thresholding algorithm.
            # - cv2.THRESH_BINARY: The thresholding type.
            # - 11: The block size (neighborhood area).
            # - 2: The constant subtracted from the mean.
            binary_image = cv2.adaptiveThreshold(gray_image, 255,
                                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)

            if self.invert_var.get():
                binary_image = cv2.bitwise_not(binary_image)

            # Convert back to PIL image
            self.binarized_image = Image.fromarray(binary_image)

            # Update the canvas to show the binarized image
            self.update_grid()

    def generate_textures(self):
        """Generates textures for the binarized image."""
        if self.binarized_image:
            # Convert the binarized image to a color image
            color_image = cv2.cvtColor(np.array(self.binarized_image), cv2.COLOR_GRAY2RGB)

            # Define colors for land and water
            land_color = [0, 255, 0]  # Green
            water_color = [255, 0, 0]  # Blue

            # Create a mask for land (white pixels in the binarized image)
            land_mask = np.all(color_image == [255, 255, 255], axis=-1)

            # Apply colors to the image
            color_image[land_mask] = land_color
            color_image[~land_mask] = water_color

            # Convert back to PIL image
            self.textured_image = Image.fromarray(color_image)

            # Update the canvas to show the textured image
            self.update_grid()

    def apply_rules(self):
        """Applies rules to the textured image."""
        if self.textured_image:
            # Convert the textured image to an array
            image_array = np.array(self.textured_image)

            # Define colors
            land_color = [0, 255, 0]  # Green
            water_color = [255, 0, 0]  # Blue
            beach_color = [0, 255, 255]  # Yellow

            # Create a copy of the image to modify
            new_image_array = image_array.copy()

            # Iterate over each pixel
            for y in range(image_array.shape[0]):
                for x in range(image_array.shape[1]):
                    # Check if the pixel is land
                    if np.array_equal(image_array[y, x], land_color):
                        # Check neighbors
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dx == 0 and dy == 0:
                                    continue
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < image_array.shape[0] and 0 <= nx < image_array.shape[1]:
                                    if np.array_equal(image_array[ny, nx], water_color):
                                        new_image_array[y, x] = beach_color
                                        break
                            else:
                                continue
                            break

            # Convert back to PIL image
            self.textured_image = Image.fromarray(new_image_array)

            # Update the canvas
            self.update_grid()

    def update_grid(self, event=None):
        self.canvas.delete("all") # Clear previous drawings

        # Determine which image to display
        if self.textured_image:
            image_to_display = self.textured_image
        elif self.binarized_image:
            image_to_display = self.binarized_image
        else:
            image_to_display = self.original_image

        if image_to_display:
            self.grid_cells = self.grid_slider.get()
            self.grid_shape = self.shape_var.get()

            # Resize image to fit canvas
            canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 600
            canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 400

            img_width, img_height = image_to_display.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            resized_image = image_to_display.resize((new_width, new_height), Image.LANCZOS)
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
        """Draws a hexagonal grid over the image with proper tessellation."""

        # Use the original approach but with corrected geometry
        hex_count_in_row = self.grid_cells

        # For pointy-top hexagons:
        # hex_width = sqrt(3) * side_length
        # hex_height = 2 * side_length
        hex_width = width / hex_count_in_row
        side_length = hex_width / np.sqrt(3)
        hex_height = 2 * side_length

        # Correct spacing for proper tessellation
        horizontal_spacing = hex_width
        vertical_spacing = hex_height * 0.75   # 3/4 of height

        # Calculate number of rows needed
        num_rows = int(height / vertical_spacing) + 2

        # Start from top-left, slightly outside to ensure coverage
        start_x = x_offset
        start_y = y_offset

        for row in range(num_rows):
            # Calculate y position for this row
            center_y = start_y + row * vertical_spacing

            # Stop if we're too far below the image
            if center_y > y_offset + height + hex_height/2:
                break

            # Stagger odd rows by half horizontal spacing
            row_offset_x = hex_width/2 if row % 2 == 1 else 0

            # Calculate how many hexagons fit in this row
            effective_width = width + hex_width  # Add some buffer
            hex_count_this_row = int(effective_width / horizontal_spacing) + 1

            for col in range(hex_count_this_row):
                # Calculate x position for this hexagon
                center_x = start_x + col * horizontal_spacing + row_offset_x

                # Skip if completely outside image bounds
                if (center_x < x_offset - hex_width/2 or
                    center_x > x_offset + width + hex_width/2 or
                    center_y < y_offset - hex_height/2 or
                    center_y > y_offset + height + hex_height/2):
                    continue

                # Draw the hexagon
                points = []
                for i in range(6):
                    angle_rad = np.deg2rad(60 * i + 30)  # +30 for pointy-top
                    x = center_x + side_length * np.cos(angle_rad)
                    y = center_y + side_length * np.sin(angle_rad)
                    points.append((x, y))

                self.canvas.create_polygon(points, outline="red", width=1, fill="")

    def on_mouse_press(self, event):
        """Records the starting coordinates of the selection."""
        self.selection_start_x = event.x
        self.selection_start_y = event.y
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        self.selection_rect = None

    def on_mouse_drag(self, event):
        """Draws the selection rectangle."""
        if self.selection_rect:
            self.canvas.delete(self.selection_rect)
        self.selection_rect = self.canvas.create_rectangle(
            self.selection_start_x, self.selection_start_y, event.x, event.y, outline="blue", width=2
        )

    def on_mouse_release(self, event):
        """Creates a new window with the selected region."""
        x1 = min(self.selection_start_x, event.x)
        y1 = min(self.selection_start_y, event.y)
        x2 = max(self.selection_start_x, event.x)
        y2 = max(self.selection_start_y, event.y)

        if self.display_image and (x2 - x1 > 0 and y2 - y1 > 0):
            # Get the image displayed on the canvas
            image_to_crop = self.textured_image or self.binarized_image or self.original_image

            # Calculate crop coordinates relative to the original image size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_width, img_height = image_to_crop.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            img_x_offset = (canvas_width - new_width) / 2
            img_y_offset = (canvas_height - new_height) / 2

            crop_x1 = (x1 - img_x_offset) / ratio
            crop_y1 = (y1 - img_y_offset) / ratio
            crop_x2 = (x2 - img_x_offset) / ratio
            crop_y2 = (y2 - img_y_offset) / ratio

            cropped_image = image_to_crop.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Create a new window to display the cropped image
            top = tk.Toplevel(self.master)
            top.title("Selected Region")
            canvas = tk.Canvas(top, width=cropped_image.width, height=cropped_image.height)
            canvas.pack()
            photo = ImageTk.PhotoImage(image=cropped_image, master=top)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo
