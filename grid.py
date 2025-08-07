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
        self.display_image = None
        self.grid_cells = 10  # Default number of grid cells
        self.grid_shape = "Square"  # Default grid shape

        # UI Elements
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(pady=10)

        self.load_button = tk.Button(self.top_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.binarize_button = tk.Button(self.top_frame, text="Binarize Image", command=self.binarize_image)
        self.binarize_button.pack(side=tk.LEFT, padx=5)

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
            self.binarized_image = None # Reset binarized image on new load
            self.display_image = None  # Clear previous display image
            self.update_grid() # Call update_grid to display the loaded image

    def binarize_image(self):
        """Converts the original image to a black and white image."""
        if self.original_image:
            # Convert PIL image to OpenCV format
            cv_image = np.array(self.original_image.convert('RGB'))
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

            # Apply binary thresholding
            # All pixels with intensity > 127 are set to 255 (white), others to 0 (black)
            _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            # Convert back to PIL image
            self.binarized_image = Image.fromarray(binary_image)

            # Update the canvas to show the binarized image
            self.update_grid()

    def update_grid(self, event=None):
        self.canvas.delete("all") # Clear previous drawings

        # Determine which image to display
        image_to_display = self.binarized_image if self.binarized_image else self.original_image

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
        """Draws a hexagonal grid over the image with proper positioning."""
        
        # Calculate hexagon dimensions based on grid_cells
        # The grid_cells value determines how many hexagons fit horizontally in the widest row
        hex_count_horizontal = self.grid_cells
        
        # Calculate hexagon size to fit the specified number horizontally
        # For a pointy-top hexagon, the width is 2 * side_length
        hex_width = width / hex_count_horizontal
        side_length = hex_width / 2
        
        # Height of a pointy-top hexagon is sqrt(3) * side_length
        hex_height = np.sqrt(3) * side_length
        
        # Vertical distance between row centers (3/4 of hex height for proper tessellation)
        row_spacing = hex_height * 0.75
        
        # Calculate number of rows that fit in the image height
        num_rows = max(1, int((height - hex_height) / row_spacing) + 1)
        
        # Calculate total grid height and center it vertically
        total_grid_height = (num_rows - 1) * row_spacing + hex_height
        grid_y_start = y_offset + (height - total_grid_height) / 2
        
        # Draw hexagons row by row
        for row in range(num_rows):
            # Calculate y position for this row
            row_y = grid_y_start + row * row_spacing + hex_height / 2
            
            # Determine horizontal offset and count for this row
            if row % 2 == 0:
                # Even rows: no offset, full count
                row_x_offset = 0
                hex_count_this_row = hex_count_horizontal
            else:
                # Odd rows: offset by half hex width, one less hexagon
                row_x_offset = hex_width / 2
                hex_count_this_row = hex_count_horizontal - 1
            
            # Calculate starting x position for this row
            row_x_start = x_offset + row_x_offset + hex_width / 2
            
            # Draw hexagons in this row
            for col in range(hex_count_this_row):
                center_x = row_x_start + col * hex_width
                center_y = row_y
                
                # Generate hexagon vertices (pointy-top orientation)
                points = []
                for i in range(6):
                    # Start at top point (-90 degrees) and go clockwise
                    angle_rad = np.deg2rad(60 * i - 90)
                    x = center_x + side_length * np.cos(angle_rad)
                    y = center_y + side_length * np.sin(angle_rad)
                    points.extend([x, y])  # Flatten coordinates for tkinter
                
                # Draw the hexagon
                self.canvas.create_polygon(points, outline="red", width=1, fill="")
                
                # Optional: Add coordinate labels for debugging
                # self.canvas.create_text(center_x, center_y, text=f"{row},{col}", 
                #                        font=("Arial", 8), fill="blue")
    # def _draw_hexagon_grid(self, width, height, x_offset, y_offset):
    #     """Draws a hexagonal grid over the image with proper tessellation."""
        
    #     # Use the original approach but with corrected geometry
    #     hex_count_in_row = self.grid_cells
        
    #     # For pointy-top hexagons:
    #     # hex_width = sqrt(3) * side_length
    #     # hex_height = 2 * side_length
    #     hex_width = width / hex_count_in_row
    #     side_length = hex_width / np.sqrt(3)
    #     hex_height = 2 * side_length
        
    #     # Correct spacing for proper tessellation
    #     horizontal_spacing = hex_width * 0.75  # 3/4 of width
    #     vertical_spacing = hex_height * 0.75   # 3/4 of height
        
    #     # Calculate number of rows needed
    #     num_rows = int(height / vertical_spacing) + 2
        
    #     # Start from top-left, slightly outside to ensure coverage
    #     start_x = x_offset
    #     start_y = y_offset
        
    #     for row in range(num_rows):
    #         # Calculate y position for this row
    #         center_y = start_y + row * vertical_spacing
            
    #         # Stop if we're too far below the image
    #         if center_y > y_offset + height + hex_height/2:
    #             break
                
    #         # Stagger odd rows by half horizontal spacing
    #         row_offset_x = horizontal_spacing / 2 if row % 2 == 1 else 0
            
    #         # Calculate how many hexagons fit in this row
    #         effective_width = width + hex_width  # Add some buffer
    #         hex_count_this_row = int(effective_width / horizontal_spacing) + 1
            
    #         for col in range(hex_count_this_row):
    #             # Calculate x position for this hexagon
    #             center_x = start_x + col * horizontal_spacing + row_offset_x
                
    #             # Skip if completely outside image bounds
    #             if (center_x < x_offset - hex_width/2 or 
    #                 center_x > x_offset + width + hex_width/2 or
    #                 center_y < y_offset - hex_height/2 or 
    #                 center_y > y_offset + height + hex_height/2):
    #                 continue
                
    #             # Draw the hexagon
    #             points = []
    #             for i in range(6):
    #                 angle_rad = np.deg2rad(60 * i + 30)  # +30 for pointy-top
    #                 x = center_x + side_length * np.cos(angle_rad)
    #                 y = center_y + side_length * np.sin(angle_rad)
    #                 points.append((x, y))
                
    #             self.canvas.create_polygon(points, outline="red", width=1, fill="")

root = tk.Tk()
app = GridGenerator(root)
root.mainloop()
