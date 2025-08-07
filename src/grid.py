import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from . import image_processing
from .enhanced_map_border import EnhancedMapBorder
from . import fantasy_terrain_drawer

class GridGenerator:
    def __init__(self, master):
        self.master = master
        master.title("Cartography Grid Generator")
        master.geometry("1200x800")
        
        # Configure grid weights for responsive layout
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(1, weight=1)

        self.image_path = None
        self.original_image = None
        self.binarized_image = None
        self.textured_image = None
        self.display_image = None
        self.grid_cells = 10
        self.grid_shape = "Square"

        # --- Selection Rectangle ---
        self.selection_rect = None
        self.selection_start_x = 0
        self.selection_start_y = 0

        self.setup_menu()
        self.setup_toolbar()
        self.setup_main_area()
        self.setup_properties_panel()
        self.setup_status_bar()

    def setup_menu(self):
        """Create a proper menu bar"""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image...", command=self.load_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Map...", command=self.save_map, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="New World Map", command=self.new_world_map)
        file_menu.add_command(label="Generate Border Map", command=self.generate_border)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset Map", command=self.reset_map)
        edit_menu.add_command(label="Clear Grid", command=self.clear_grid)
        
        # Map menu
        map_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Map", menu=map_menu)
        map_menu.add_command(label="Binarize Image", command=self.binarize_image)
        map_menu.add_command(label="Generate Textures", command=self.generate_textures)
        map_menu.add_command(label="Apply Rules", command=self.apply_rules)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        self.show_grid_var = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Show Grid", variable=self.show_grid_var, command=self.update_grid)
        self.show_coordinates_var = tk.BooleanVar(value=False)
        view_menu.add_checkbutton(label="Show Coordinates", variable=self.show_coordinates_var, command=self.update_grid)
        
        # Bind keyboard shortcuts
        self.master.bind("<Control-o>", lambda e: self.load_image())
        self.master.bind("<Control-s>", lambda e: self.save_map())

    def setup_toolbar(self):
        """Create a toolbar with commonly used functions"""
        toolbar = tk.Frame(self.master, relief=tk.RAISED, bd=1)
        toolbar.grid(row=0, column=0, columnspan=3, sticky="ew", padx=2, pady=2)
        
        # Quick action buttons
        buttons = [
            ("Load Image", self.load_image, "📁"),
            ("Binarize", self.binarize_image, "🔲"),
            ("Textures", self.generate_textures, "🎨"),
            ("Rules", self.apply_rules, "📏"),
            ("Border", self.generate_border, "🗺️")
        ]
        
        for i, (text, command, emoji) in enumerate(buttons):
            btn = tk.Button(toolbar, text=f"{emoji} {text}", command=command, 
                          relief=tk.FLAT, padx=10, pady=5)
            btn.pack(side=tk.LEFT, padx=2, pady=2)
            
        # Separator
        separator = ttk.Separator(toolbar, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Mode selection
        tk.Label(toolbar, text="Mode:").pack(side=tk.LEFT, padx=(10, 5))
        self.mode_var = tk.StringVar(value="World Map")
        mode_combo = ttk.Combobox(toolbar, textvariable=self.mode_var, 
                                 values=["World Map", "Simple Map"], 
                                 state="readonly", width=12)
        mode_combo.pack(side=tk.LEFT, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)

    def setup_main_area(self):
        """Setup the main canvas area"""
        main_frame = tk.Frame(self.master)
        main_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas with scrollbars
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", scrollregion=(0, 0, 1000, 1000))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Mouse bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

    def setup_properties_panel(self):
        """Setup the properties panel on the right"""
        props_frame = tk.LabelFrame(self.master, text="Properties", padx=10, pady=10)
        props_frame.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)
        
        # Grid controls
        grid_frame = tk.LabelFrame(props_frame, text="Grid Settings")
        grid_frame.pack(fill="x", pady=5)
        
        tk.Label(grid_frame, text="Grid Cells:").pack(anchor="w")
        self.grid_slider = tk.Scale(grid_frame, from_=1, to_=100, orient=tk.HORIZONTAL,
                                   command=self.update_grid, length=200)
        self.grid_slider.set(self.grid_cells)
        self.grid_slider.pack(fill="x", pady=2)
        
        tk.Label(grid_frame, text="Grid Shape:").pack(anchor="w", pady=(10, 0))
        self.shape_var = tk.StringVar(value=self.grid_shape)
        shape_combo = ttk.Combobox(grid_frame, textvariable=self.shape_var,
                                  values=["Square", "Hexagon"], state="readonly")
        shape_combo.pack(fill="x", pady=2)
        shape_combo.bind("<<ComboboxSelected>>", lambda e: self.update_grid())
        
        # Image processing controls
        process_frame = tk.LabelFrame(props_frame, text="Image Processing")
        process_frame.pack(fill="x", pady=5)
        
        self.invert_var = tk.BooleanVar()
        invert_check = tk.Checkbutton(process_frame, text="Invert Binarization", 
                                     var=self.invert_var, command=self.binarize_image)
        invert_check.pack(anchor="w")
        
        self.fantasy_var = tk.BooleanVar()
        fantasy_check = tk.Checkbutton(process_frame, text="Fantasy Style", 
                                      var=self.fantasy_var, command=self.update_grid)
        fantasy_check.pack(anchor="w")
        
        # Map information
        info_frame = tk.LabelFrame(props_frame, text="Map Information")
        info_frame.pack(fill="x", pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, wrap=tk.WORD)
        self.info_text.pack(fill="both", expand=True)
        info_scroll = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)

    def setup_status_bar(self):
        """Setup status bar at the bottom"""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.master, textvariable=self.status_var, 
                            relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=2, column=0, columnspan=3, sticky="ew")

    def update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.master.update_idletasks()

    def update_info(self):
        """Update the information panel"""
        info = []
        if self.original_image:
            info.append(f"Original Size: {self.original_image.size}")
        if self.binarized_image:
            info.append("Status: Binarized")
        if self.textured_image:
            info.append("Status: Textured")
        
        info.append(f"Grid: {self.grid_shape} ({self.grid_cells} cells)")
        info.append(f"Mode: {self.mode_var.get()}")
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, "\n".join(info))

    # New methods for menu functionality
    def save_map(self):
        """Save the current map"""
        if self.textured_image or self.binarized_image or self.original_image:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if file_path:
                image_to_save = self.textured_image or self.binarized_image or self.original_image
                image_to_save.save(file_path)
                self.update_status(f"Map saved to {file_path}")
        else:
            messagebox.showwarning("No Map", "No map to save. Please load or generate a map first.")

    def new_world_map(self):
        """Create a new world map"""
        self.reset_map()
        self.update_status("Ready to create new world map")

    def reset_map(self):
        """Reset all map data"""
        self.original_image = None
        self.binarized_image = None
        self.textured_image = None
        self.display_image = None
        self.canvas.delete("all")
        self.update_info()
        self.update_status("Map reset")

    def clear_grid(self):
        """Clear only the grid overlay"""
        self.canvas.delete("grid_line")
        self.update_status("Grid cleared")

    def on_mode_change(self, event):
        """Handle mode change"""
        mode = self.mode_var.get()
        if mode == "Simple Map":
            self.update_status("Simple Map mode: Click and drag to select a region")
        else:
            self.update_status("World Map mode")

    def on_mousewheel(self, event):
        """Handle mouse wheel for zooming"""
        # Basic zoom functionality could be added here
        pass

    def on_canvas_configure(self, event):
        """Handle canvas resize"""
        self.update_grid()

    # Original methods with updates
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Load Map Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.image_path = file_path
                self.original_image = Image.open(self.image_path)
                self.binarized_image = None
                self.textured_image = None
                self.display_image = None
                self.update_grid()
                self.update_info()
                self.update_status(f"Loaded: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def binarize_image(self):
        """Converts the original image to a black and white image."""
        if self.original_image:
            try:
                self.binarized_image = image_processing.binarize_image(
                    self.original_image, self.invert_var.get()
                )
                self.update_grid()
                self.update_info()
                self.update_status("Image binarized")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to binarize image: {str(e)}")
        else:
            messagebox.showwarning("No Image", "Please load an image first.")

    def generate_textures(self):
        """Generates textures for the binarized image."""
        if self.binarized_image:
            try:
                if self.fantasy_var.get():
                    self.update_status("Generating fantasy terrain...")
                    base_mask = np.array(self.binarized_image)
                    # Ensure mask is binary (0 or 255)
                    base_mask[base_mask > 0] = 255
                    width, height = self.binarized_image.size
                    fantasy_map_array = fantasy_terrain_drawer.create_fantasy_terrain_overlay(base_mask, width, height)
                    self.textured_image = Image.fromarray(fantasy_map_array)
                    self.update_status("Fantasy terrain generated")
                else:
                    self.textured_image = image_processing.generate_textures(self.binarized_image)
                    self.update_status("Textures generated")

                self.update_grid()
                self.update_info()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate textures: {str(e)}")
        else:
            messagebox.showwarning("No Binarized Image", "Please binarize an image first.")

    def apply_rules(self):
        """Applies rules to the textured image."""
        if self.textured_image:
            try:
                self.textured_image = image_processing.apply_rules(self.textured_image)
                self.update_grid()
                self.update_info()
                self.update_status("Rules applied")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to apply rules: {str(e)}")
        else:
            messagebox.showwarning("No Textured Image", "Please generate textures first.")

    def generate_border(self):
        """Generates a new map with a natural border."""
        try:
            width = max(800, self.canvas.winfo_width())
            height = max(600, self.canvas.winfo_height())
            border_generator = EnhancedMapBorder(width, height)
            border_mask = border_generator.generate_continent()
            self.original_image = Image.fromarray(border_mask)
            self.binarized_image = self.original_image
            self.textured_image = None
            self.update_grid()
            self.update_info()
            self.update_status("Border map generated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate border: {str(e)}")

    def update_grid(self, event=None):
        """Update the grid display"""
        if not self.show_grid_var.get():
            self.canvas.delete("grid_line")
            return
            
        self.canvas.delete("all")

        # Determine which image to display
        image_to_display = self.textured_image or self.binarized_image or self.original_image

        if image_to_display:
            self.grid_cells = self.grid_slider.get()
            self.grid_shape = self.shape_var.get()

            # Resize image to fit canvas
            canvas_width = max(self.canvas.winfo_width(), 600)
            canvas_height = max(self.canvas.winfo_height(), 400)

            img_width, img_height = image_to_display.size
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)

            resized_image = image_to_display.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(canvas_width / 2, canvas_height / 2, 
                                   image=self.display_image, anchor=tk.CENTER)

            # Calculate image offset
            img_x_offset = (canvas_width - new_width) / 2
            img_y_offset = (canvas_height - new_height) / 2

            if self.show_grid_var.get():
                if self.grid_shape == "Square":
                    self._draw_square_grid(new_width, new_height, img_x_offset, img_y_offset)
                elif self.grid_shape == "Hexagon":
                    self._draw_hexagon_grid(new_width, new_height, img_x_offset, img_y_offset)

        self.update_info()

    def _draw_square_grid(self, width, height, x_offset, y_offset):
        """Draws a square grid over the image."""
        cell_size_x = width / self.grid_cells
        cell_size_y = height / self.grid_cells
        
        for i in range(self.grid_cells + 1):
            # Vertical lines
            x = x_offset + i * cell_size_x
            self.canvas.create_line(x, y_offset, x, y_offset + height, 
                                  fill="red", width=1, tags="grid_line")
        
        for j in range(self.grid_cells + 1):
            # Horizontal lines
            y = y_offset + j * cell_size_y
            self.canvas.create_line(x_offset, y, x_offset + width, y, 
                                  fill="red", width=1, tags="grid_line")
        
        # Add coordinates if enabled
        if self.show_coordinates_var.get():
            for i in range(self.grid_cells):
                for j in range(self.grid_cells):
                    x = x_offset + (i + 0.5) * cell_size_x
                    y = y_offset + (j + 0.5) * cell_size_y
                    self.canvas.create_text(x, y, text=f"{i},{j}", fill="blue", 
                                          font=("Arial", 8), tags="grid_line")

    def _draw_hexagon_grid(self, width, height, x_offset, y_offset):
        """Draws a hexagonal grid over the image with proper tessellation."""
        hex_count_in_row = self.grid_cells
        hex_width = width / hex_count_in_row
        side_length = hex_width / np.sqrt(3)
        hex_height = 2 * side_length
        
        horizontal_spacing = hex_width
        vertical_spacing = hex_height * 0.75
        
        num_rows = int(height / vertical_spacing) + 2
        start_x = x_offset
        start_y = y_offset

        for row in range(num_rows):
            center_y = start_y + row * vertical_spacing
            if center_y > y_offset + height + hex_height/2:
                break

            row_offset_x = hex_width/2 if row % 2 == 1 else 0
            effective_width = width + hex_width
            hex_count_this_row = int(effective_width / horizontal_spacing) + 1

            for col in range(hex_count_this_row):
                center_x = start_x + col * horizontal_spacing + row_offset_x
                
                if (center_x < x_offset - hex_width/2 or
                    center_x > x_offset + width + hex_width/2 or
                    center_y < y_offset - hex_height/2 or
                    center_y > y_offset + height + hex_height/2):
                    continue

                points = []
                for i in range(6):
                    angle_rad = np.deg2rad(60 * i + 30)
                    x = center_x + side_length * np.cos(angle_rad)
                    y = center_y + side_length * np.sin(angle_rad)
                    points.append((x, y))

                self.canvas.create_polygon(points, outline="red", width=1, fill="", 
                                         tags="grid_line")
                
                # Add coordinates if enabled
                if self.show_coordinates_var.get():
                    self.canvas.create_text(center_x, center_y, text=f"{col},{row}", 
                                          fill="blue", font=("Arial", 8), tags="grid_line")

    def on_mouse_press(self, event):
        """Records the starting coordinates of the selection."""
        if self.mode_var.get() == "Simple Map":
            self.selection_start_x = event.x
            self.selection_start_y = event.y
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            self.selection_rect = None

    def on_mouse_drag(self, event):
        """Draws the selection rectangle."""
        if self.mode_var.get() == "Simple Map":
            if self.selection_rect:
                self.canvas.delete(self.selection_rect)
            self.selection_rect = self.canvas.create_rectangle(
                self.selection_start_x, self.selection_start_y, event.x, event.y, 
                outline="blue", width=2, dash=(5, 5)
            )

    def on_mouse_release(self, event):
        """Creates a new window with the selected region."""
        if self.mode_var.get() == "Simple Map":
            x1 = min(self.selection_start_x, event.x)
            y1 = min(self.selection_start_y, event.y)
            x2 = max(self.selection_start_x, event.x)
            y2 = max(self.selection_start_y, event.y)

            if self.display_image and (x2 - x1 > 10 and y2 - y1 > 10):
                self.create_simple_map_window(x1, y1, x2, y2)

    def create_simple_map_window(self, x1, y1, x2, y2):
        """Create a new window for the simple map"""
        image_to_crop = self.textured_image or self.binarized_image or self.original_image
        
        # Calculate crop coordinates
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = image_to_crop.size
        ratio = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        img_x_offset = (canvas_width - new_width) / 2
        img_y_offset = (canvas_height - new_height) / 2

        crop_x1 = max(0, (x1 - img_x_offset) / ratio)
        crop_y1 = max(0, (y1 - img_y_offset) / ratio)
        crop_x2 = min(img_width, (x2 - img_x_offset) / ratio)
        crop_y2 = min(img_height, (y2 - img_y_offset) / ratio)

        cropped_image = image_to_crop.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # Create new window with the cropped region
        top = tk.Toplevel(self.master)
        top.title("Simple Map - Selected Region")
        top.geometry("800x600")
        
        # Create a new GridGenerator instance for the simple map
        simple_map_app = GridGenerator(top)
        simple_map_app.original_image = cropped_image
        simple_map_app.mode_var.set("Simple Map")
        simple_map_app.update_grid()
        simple_map_app.update_status("Simple map created from selection")