import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import random
from typing import List, Tuple, Optional, Dict
import math

class FantasyTerrainArtist:
    """
    Creates fantasy map terrain features with hand-drawn, pencil-sketched aesthetics.
    Inspired by classic D&D and fantasy cartography styles.
    """
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Colors for different terrain types (RGB)
        self.terrain_colors = {
            'land': (245, 245, 220),      # Beige
            'water': (176, 224, 230),      # Light blue
            'forest': (34, 139, 34),       # Forest green
            'mountains': (139, 137, 137),   # Dark gray
            'hills': (205, 205, 180),      # Light olive
            'desert': (238, 203, 173),     # Peach
            'swamp': (107, 142, 35),       # Olive drab
            'snow': (255, 250, 250),       # Snow white
            'roads': (139, 69, 19),        # Saddle brown
            'settlements': (0, 0, 0)        # Black
        }

        # Line weights for different features
        self.line_weights = {
            'coastline': 2,
            'rivers': 2,
            'roads': 1,
            'borders': 1,
            'contours': 1
        }

    def create_fantasy_map(self, base_mask: np.ndarray, terrain_density: float = 0.7) -> np.ndarray:
        """Create a complete fantasy map with various terrain features."""
        # Start with base colors
        map_image = self._create_base_map(base_mask)

        # Add terrain types based on elevation and proximity
        terrain_map = self._generate_terrain_types(base_mask)
        map_image = self._apply_terrain_colors(map_image, terrain_map)

        # Add hand-drawn style features
        map_image = self._add_mountain_ranges(map_image, terrain_map)
        map_image = self._add_forest_areas(map_image, terrain_map)
        map_image = self._add_hills_and_elevation(map_image, terrain_map)
        map_image = self._add_rivers_and_streams(map_image, base_mask)
        map_image = self._add_roads_and_paths(map_image, base_mask)
        map_image = self._add_settlements(map_image, base_mask)
        map_image = self._add_decorative_elements(map_image)

        # Apply hand-drawn styling effects
        map_image = self._apply_sketchy_effects(map_image)

        return map_image

    def _create_base_map(self, base_mask: np.ndarray) -> np.ndarray:
        """Create the base map with land and water."""
        map_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Fill water areas
        map_image[base_mask == 0] = self.terrain_colors['water']
        # Fill land areas
        map_image[base_mask == 255] = self.terrain_colors['land']

        return map_image

    def _generate_terrain_types(self, base_mask: np.ndarray) -> np.ndarray:
        """Generate different terrain types based on various factors."""
        terrain_map = np.zeros((self.height, self.width), dtype=np.uint8)
        land_mask = base_mask == 255

        # Generate elevation map using multiple noise layers
        elevation = self._generate_elevation_map()

        # Generate moisture map
        moisture = self._generate_moisture_map(base_mask)

        # Generate temperature map (latitude-based with variations)
        temperature = self._generate_temperature_map()

        # Assign terrain types based on elevation, moisture, and temperature
        for y in range(self.height):
            for x in range(self.width):
                if land_mask[y, x]:
                    elev = elevation[y, x]
                    moist = moisture[y, x]
                    temp = temperature[y, x]

                    terrain_map[y, x] = self._classify_terrain(elev, moist, temp)

        return terrain_map

    def _classify_terrain(self, elevation: float, moisture: float, temperature: float) -> int:
        """Classify terrain type based on environmental factors."""
        # Terrain type codes
        PLAINS = 1
        FOREST = 2
        HILLS = 3
        MOUNTAINS = 4
        DESERT = 5
        SWAMP = 6
        SNOW = 7

        # High elevation = mountains or snow
        if elevation > 0.8:
            return SNOW if temperature < 0.3 else MOUNTAINS
        elif elevation > 0.6:
            return HILLS

        # Low moisture areas
        if moisture < 0.3:
            return DESERT

        # High moisture, low elevation
        if moisture > 0.8 and elevation < 0.3:
            return SWAMP

        # Forest conditions
        if moisture > 0.5 and temperature > 0.3 and temperature < 0.8:
            return FOREST

        # Default to plains
        return PLAINS

    def _generate_elevation_map(self) -> np.ndarray:
        """Generate elevation map using multiple noise octaves."""
        elevation = np.zeros((self.height, self.width))

        # Multiple octaves for realistic elevation
        frequencies = [0.01, 0.02, 0.04, 0.08]
        amplitudes = [1.0, 0.5, 0.25, 0.125]

        for freq, amp in zip(frequencies, amplitudes):
            noise = np.random.random((self.height, self.width))
            noise = gaussian_filter(noise, sigma=1/freq)
            elevation += amp * noise

        # Normalize
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        return elevation

    def _generate_moisture_map(self, base_mask: np.ndarray) -> np.ndarray:
        """Generate moisture map based on distance from water."""
        # Distance from water bodies
        water_mask = (base_mask == 0).astype(np.uint8)
        distance_from_water = cv2.distanceTransform(1 - water_mask, cv2.DIST_L2, 5)

        # Normalize and invert (closer to water = more moisture)
        max_dist = distance_from_water.max()
        if max_dist > 0:
            moisture_from_water = 1 - (distance_from_water / max_dist)
        else:
            moisture_from_water = np.zeros_like(distance_from_water, dtype=float)


        # Add noise for variation
        noise = np.random.random((self.height, self.width))
        noise = gaussian_filter(noise, sigma=20)

        # Combine
        moisture = 0.7 * moisture_from_water + 0.3 * noise
        return np.clip(moisture, 0, 1)

    def _generate_temperature_map(self) -> np.ndarray:
        """Generate temperature map with latitude effects and local variations."""
        temperature = np.zeros((self.height, self.width))

        # Latitude effect (cooler toward poles - top and bottom of map)
        for y in range(self.height):
            lat_factor = 1 - 2 * abs(y - self.height/2) / self.height
            temperature[y, :] = 0.5 + 0.4 * lat_factor

        # Add local variations
        noise = np.random.random((self.height, self.width))
        noise = gaussian_filter(noise, sigma=30)
        temperature += 0.3 * (noise - 0.5)

        return np.clip(temperature, 0, 1)

    def _apply_terrain_colors(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Apply colors based on terrain types."""
        color_mapping = {
            1: self.terrain_colors['land'],     # Plains
            2: self.terrain_colors['forest'],   # Forest
            3: self.terrain_colors['hills'],    # Hills
            4: self.terrain_colors['mountains'], # Mountains
            5: self.terrain_colors['desert'],   # Desert
            6: self.terrain_colors['swamp'],    # Swamp
            7: self.terrain_colors['snow']      # Snow
        }

        for terrain_type, color in color_mapping.items():
            mask = terrain_map == terrain_type
            map_image[mask] = color

        return map_image

    def _add_mountain_ranges(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Add hand-drawn mountain symbols."""
        mountain_mask = (terrain_map == 4) | (terrain_map == 7)  # Mountains or snow
        mountain_points = np.where(mountain_mask)

        if len(mountain_points[0]) > 0:
            # Convert to PIL for easier drawing
            pil_image = Image.fromarray(map_image)
            draw = ImageDraw.Draw(pil_image)
            
            # Group mountain points into ranges
            mountain_ranges = self._group_nearby_points(mountain_points, distance_threshold=30)
            
            for mountain_range in mountain_ranges:
                for y, x in mountain_range:
                    self._draw_mountain_symbol(draw, x, y)
            
            map_image = np.array(pil_image)
        
        return map_image
    
    def _draw_mountain_symbol(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a single hand-drawn mountain symbol."""
        # Random mountain size and shape
        base_width = np.random.randint(8, 15)
        height = np.random.randint(10, 18)
        
        # Create jagged mountain outline
        peak_x = x + np.random.randint(-3, 4)
        peak_y = y - height
        
        # Left slope
        left_points = []
        left_base_x = x - base_width // 2
        steps = np.random.randint(3, 6)
        for i in range(steps):
            step_x = left_base_x + (peak_x - left_base_x) * i / (steps - 1)
            step_y = y - height * (i / (steps - 1)) ** 0.7  # Curved slope
            step_x += np.random.uniform(-2, 2)  # Add randomness
            step_y += np.random.uniform(-1, 1)
            left_points.append((step_x, step_y))
        
        # Right slope
        right_points = []
        right_base_x = x + base_width // 2
        for i in range(steps-1, -1, -1):
            step_x = right_base_x + (peak_x - right_base_x) * i / (steps - 1)
            step_y = y - height * (i / (steps - 1)) ** 0.7
            step_x += np.random.uniform(-2, 2)
            step_y += np.random.uniform(-1, 1)
            right_points.append((step_x, step_y))
        
        # Complete mountain outline
        mountain_points = [(left_base_x, y)] + left_points + [(peak_x, peak_y)] + right_points + [(right_base_x, y)]
        
        # Draw mountain with sketchy lines
        for i in range(len(mountain_points) - 1):
            self._draw_sketchy_line(draw, mountain_points[i], mountain_points[i+1], width=1)
        
        # Add shading lines
        if np.random.random() > 0.5:
            self._add_mountain_shading(draw, x, y, base_width, height)
    
    def _add_mountain_shading(self, draw: ImageDraw.Draw, x: int, y: int, width: int, height: int):
        """Add shading lines to mountain."""
        num_lines = np.random.randint(2, 5)
        for _ in range(num_lines):
            line_start_x = x - width//4 + np.random.randint(0, width//2)
            line_start_y = y - np.random.randint(height//3, height*2//3)
            line_end_x = line_start_x + np.random.randint(-5, 6)
            line_end_y = line_start_y + np.random.randint(3, 8)
            
            self._draw_sketchy_line(draw, (line_start_x, line_start_y), (line_end_x, line_end_y), width=1)
    
    def _add_forest_areas(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Add hand-drawn forest symbols."""
        forest_mask = terrain_map == 2
        forest_points = np.where(forest_mask)
        
        if len(forest_points[0]) > 0:
            pil_image = Image.fromarray(map_image)
            draw = ImageDraw.Draw(pil_image)
            
            # Sample forest points (don't draw on every pixel)
            sample_indices = np.random.choice(len(forest_points[0]), 
                                            size=min(len(forest_points[0])//20, 200), 
                                            replace=False)
            
            for idx in sample_indices:
                y, x = forest_points[0][idx], forest_points[1][idx]
                if np.random.random() > 0.7:  # Only draw some trees
                    self._draw_tree_symbol(draw, x, y)
            
            map_image = np.array(pil_image)
        
        return map_image
    
    def _draw_tree_symbol(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a single hand-drawn tree symbol."""
        tree_styles = ['conifer', 'deciduous', 'simple']
        style = np.random.choice(tree_styles)
        
        if style == 'conifer':
            self._draw_conifer_tree(draw, x, y)
        elif style == 'deciduous':
            self._draw_deciduous_tree(draw, x, y)
        else:
            self._draw_simple_tree(draw, x, y)
    
    def _draw_conifer_tree(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a coniferous tree (triangle-based)."""
        height = np.random.randint(8, 15)
        width = np.random.randint(6, 10)
        
        # Draw triangular layers
        num_layers = np.random.randint(2, 4)
        layer_height = height // num_layers
        
        for layer in range(num_layers):
            layer_y = y - layer * layer_height * 0.7
            layer_width = width * (1 - layer * 0.2)
            
            # Draw triangle with slight randomness
            points = [
                (x, layer_y - layer_height),  # Top
                (x - layer_width//2, layer_y),  # Bottom left
                (x + layer_width//2, layer_y)   # Bottom right
            ]
            
            # Add randomness to points
            points = [(px + np.random.uniform(-1, 1), py + np.random.uniform(-1, 1)) 
                     for px, py in points]
            
            # Draw with sketchy lines
            for i in range(len(points)):
                next_i = (i + 1) % len(points)
                self._draw_sketchy_line(draw, points[i], points[next_i], width=1)
        
        # Draw trunk
        trunk_bottom = y + 2
        trunk_top = y - 2
        self._draw_sketchy_line(draw, (x, trunk_top), (x, trunk_bottom), width=2)
    
    def _draw_deciduous_tree(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a deciduous tree (cloud-like crown)."""
        crown_radius = np.random.randint(5, 9)
        trunk_height = np.random.randint(6, 10)
        
        # Draw cloud-like crown with multiple circles
        num_circles = np.random.randint(3, 6)
        for _ in range(num_circles):
            circle_x = x + np.random.randint(-crown_radius//2, crown_radius//2)
            circle_y = y - crown_radius + np.random.randint(-2, 3)
            circle_r = np.random.randint(crown_radius//2, crown_radius)
            
            # Draw sketchy circle
            self._draw_sketchy_circle(draw, circle_x, circle_y, circle_r)
        
        # Draw trunk
        trunk_bottom = y + 2
        trunk_top = y - trunk_height//3
        self._draw_sketchy_line(draw, (x, trunk_top), (x, trunk_bottom), width=2)
    
    def _draw_simple_tree(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a simple tree symbol (just a circle on a line)."""
        crown_radius = np.random.randint(4, 7)
        crown_y = y - crown_radius - 2
        
        # Crown
        self._draw_sketchy_circle(draw, x, crown_y, crown_radius)
        
        # Trunk
        trunk_bottom = y + 1
        trunk_top = crown_y + crown_radius - 1
        self._draw_sketchy_line(draw, (x, trunk_top), (x, trunk_bottom), width=2)
    
    def _add_hills_and_elevation(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Add contour lines and hill symbols."""
        hills_mask = terrain_map == 3
        hills_points = np.where(hills_mask)
        
        if len(hills_points[0]) > 0:
            pil_image = Image.fromarray(map_image)
            draw = ImageDraw.Draw(pil_image)
            
            # Add some hill symbols
            sample_indices = np.random.choice(len(hills_points[0]), 
                                            size=min(len(hills_points[0])//30, 50), 
                                            replace=False)
            
            for idx in sample_indices:
                y, x = hills_points[0][idx], hills_points[1][idx]
                if np.random.random() > 0.8:
                    self._draw_hill_symbol(draw, x, y)
            
            map_image = np.array(pil_image)
        
        return map_image
    
    def _draw_hill_symbol(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a hill symbol (curved mound)."""
        width = np.random.randint(15, 25)
        height = np.random.randint(6, 12)
        
        # Create curved hill outline
        points = []
        for i in range(11):  # 11 points for smooth curve
            t = i / 10.0
            hill_x = x - width//2 + width * t
            # Parabolic curve for hill shape
            hill_y = y - height * (1 - (2*t - 1)**2)
            hill_x += np.random.uniform(-1, 1)  # Add slight randomness
            hill_y += np.random.uniform(-0.5, 0.5)
            points.append((hill_x, hill_y))
        
        # Add base points to close the shape
        points.append((x + width//2, y))
        points.append((x - width//2, y))
        
        # Draw hill outline
        for i in range(len(points) - 1):
            self._draw_sketchy_line(draw, points[i], points[i+1], width=1)
    
    def _add_rivers_and_streams(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add meandering rivers and streams."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Generate rivers from high elevation to water
        num_rivers = np.random.randint(3, 8)
        
        for _ in range(num_rivers):
            self._draw_river(draw, base_mask)
        
        return np.array(pil_image)
    
    def _draw_river(self, draw: ImageDraw.Draw, base_mask: np.ndarray):
        """Draw a single meandering river."""
        # Find a starting point on land, preferably inland
        land_points = np.where(base_mask == 255)
        if len(land_points[0]) == 0:
            return
        
        # Choose starting point
        start_idx = np.random.randint(0, len(land_points[0]))
        start_y, start_x = land_points[0][start_idx], land_points[1][start_idx]
        
        # River parameters
        current_x, current_y = start_x, start_y
        direction = np.random.uniform(0, 2*np.pi)  # Initial direction
        river_points = [(current_x, current_y)]
        
        # Generate river path
        for step in range(200):  # Max river length
            # Add meandering
            direction += np.random.normal(0, 0.3)  # Direction change
            
            # Move forward
            step_size = np.random.uniform(3, 8)
            next_x = current_x + step_size * np.cos(direction)
            next_y = current_y + step_size * np.sin(direction)
            
            # Check bounds
            if (next_x < 0 or next_x >= self.width or 
                next_y < 0 or next_y >= self.height):
                break
            
            # Check if we've reached water (end river)
            if base_mask[int(next_y), int(next_x)] == 0:
                river_points.append((next_x, next_y))
                break
            
            river_points.append((next_x, next_y))
            current_x, current_y = next_x, next_y
        
        # Draw river as connected sketchy lines
        for i in range(len(river_points) - 1):
            river_width = max(1, 3 - i // 30)  # Taper river width
            self._draw_sketchy_line(draw, river_points[i], river_points[i+1], 
                                  width=river_width, color=(100, 150, 200))
    
    def _add_roads_and_paths(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add roads and paths connecting settlements."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Generate a few main roads
        num_roads = np.random.randint(2, 5)
        
        for _ in range(num_roads):
            self._draw_road(draw, base_mask)
        
        return np.array(pil_image)
    
    def _draw_road(self, draw: ImageDraw.Draw, base_mask: np.ndarray):
        """Draw a single road path."""
        # Random start and end points on land
        land_points = np.where(base_mask == 255)
        if len(land_points[0]) < 2:
            return
        
        start_idx = np.random.randint(0, len(land_points[0]))
        end_idx = np.random.randint(0, len(land_points[0]))
        
        start_x, start_y = land_points[1][start_idx], land_points[0][start_idx]
        end_x, end_y = land_points[1][end_idx], land_points[0][end_idx]
        
        # Create path with some curvature
        road_points = self._create_curved_path(start_x, start_y, end_x, end_y)
        
        # Draw road as dashed line
        for i in range(len(road_points) - 1):
            if i % 3 == 0:  # Dashed effect
                self._draw_sketchy_line(draw, road_points[i], road_points[i+1], 
                                      width=1, color=self.terrain_colors['roads'])
    
    def _create_curved_path(self, start_x: float, start_y: float, 
                           end_x: float, end_y: float) -> List[Tuple[float, float]]:
        """Create a curved path between two points."""
        # Number of segments
        num_segments = int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / 10)
        num_segments = max(5, min(num_segments, 20))
        
        # Create control points for curve
        mid_x = (start_x + end_x) / 2 + np.random.uniform(-50, 50)
        mid_y = (start_y + end_y) / 2 + np.random.uniform(-50, 50)
        
        path_points = []
        for i in range(num_segments + 1):
            t = i / num_segments
            # Quadratic Bézier curve
            x = (1-t)**2 * start_x + 2*(1-t)*t * mid_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * mid_y + t**2 * end_y
            path_points.append((x, y))
        
        return path_points
    
    def _add_settlements(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add settlement symbols (towns, cities)."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Generate settlements
        num_settlements = np.random.randint(3, 8)
        land_points = np.where(base_mask == 255)
        
        if len(land_points[0]) > 0:
            for _ in range(num_settlements):
                # Random location on land
                idx = np.random.randint(0, len(land_points[0]))
                y, x = land_points[0][idx], land_points[1][idx]
                
                settlement_type = np.random.choice(['village', 'town', 'city'], 
                                                 p=[0.6, 0.3, 0.1])
                self._draw_settlement(draw, x, y, settlement_type)
        
        return np.array(pil_image)
    
    def _draw_settlement(self, draw: ImageDraw.Draw, x: int, y: int, settlement_type: str):
        """Draw settlement symbol based on type."""
        if settlement_type == 'village':
            # Simple house symbol
            self._draw_house_symbol(draw, x, y, size=6)
        elif settlement_type == 'town':
            # Multiple houses
            for i in range(3):
                house_x = x + np.random.randint(-8, 9)
                house_y = y + np.random.randint(-8, 9)
                self._draw_house_symbol(draw, house_x, house_y, size=5)
        else:  # city
            # Castle/fortress symbol
            self._draw_castle_symbol(draw, x, y)
    
    def _draw_house_symbol(self, draw: ImageDraw.Draw, x: int, y: int, size: int = 6):
        """Draw a simple house symbol."""
        # House base (square)
        base_points = [
            (x - size//2, y - size//2),
            (x + size//2, y - size//2),
            (x + size//2, y + size//2),
            (x - size//2, y + size//2)
        ]
        
        # Draw base
        for i in range(len(base_points)):
            next_i = (i + 1) % len(base_points)
            self._draw_sketchy_line(draw, base_points[i], base_points[next_i], width=1)
        
        # Roof (triangle)
        roof_points = [
            (x - size//2, y - size//2),
            (x, y - size),
            (x + size//2, y - size//2)
        ]
        
        for i in range(len(roof_points) - 1):
            self._draw_sketchy_line(draw, roof_points[i], roof_points[i+1], width=1)
    
    def _draw_castle_symbol(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a castle/fortress symbol."""
        size = 12
        
        # Main wall
        wall_points = [
            (x - size, y - size//2),
            (x + size, y - size//2),
            (x + size, y + size//2),
            (x - size, y + size//2)
        ]
        
        for i in range(len(wall_points)):
            next_i = (i + 1) % len(wall_points)
            self._draw_sketchy_line(draw, wall_points[i], wall_points[next_i], width=2)
        
        # Towers
        tower_positions = [(x - size, y - size//2), (x, y - size//2), (x + size, y - size//2)]
        for tower_x, tower_y in tower_positions:
            # Tower rectangle
            tower_top = tower_y - size//2
            self._draw_sketchy_line(draw, (tower_x - 3, tower_y), (tower_x - 3, tower_top), width=1)
            self._draw_sketchy_line(draw, (tower_x + 3, tower_y), (tower_x + 3, tower_top), width=1)
            self._draw_sketchy_line(draw, (tower_x - 3, tower_top), (tower_x + 3, tower_top), width=1)
            
            # Crenellations
            for i in range(-1, 2):
                cren_x = tower_x + i * 2
                self._draw_sketchy_line(draw, (cren_x, tower_top), (cren_x, tower_top - 2), width=1)
    
    def _add_decorative_elements(self, map_image: np.ndarray) -> np.ndarray:
        """Add decorative elements like compass rose, scale, etc."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Add compass rose
        if np.random.random() > 0.5:
            compass_x = self.width - 80
            compass_y = 80
            self._draw_compass_rose(draw, compass_x, compass_y)
        
        # Add decorative border elements
        if np.random.random() > 0.7:
            self._add_map_border_decoration(draw)
        
        return np.array(pil_image)
    
    def _draw_compass_rose(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a simple compass rose."""
        size = 30
        
        # Main directions
        directions = [
            (0, -size, 'N'),      # North
            (size, 0, 'E'),       # East
            (0, size, 'S'),       # South
            (-size, 0, 'W')       # West
        ]
        
        # Draw compass lines
        for dx, dy, label in directions:
            end_x, end_y = x + dx, y + dy
            self._draw_sketchy_line(draw, (x, y), (end_x, end_y), width=1)
            
            # Add direction labels
            text_x = x + dx * 1.3
            text_y = y + dy * 1.3
            try:
                draw.text((text_x, text_y), label, fill=(0, 0, 0))
            except:
                pass  # Skip if font not available
        
        # Central circle
        self._draw_sketchy_circle(draw, x, y, 5)
    
    def _add_map_border_decoration(self, draw: ImageDraw.Draw):
        """Add decorative elements to map border."""
        # Simple corner decorations
        corner_size = 20
        corners = [
            (10, 10), (self.width - 10, 10), 
            (self.width - 10, self.height - 10), (10, self.height - 10)
        ]
        
        for corner_x, corner_y in corners:
            # Simple corner flourish
            self._draw_sketchy_line(draw, (corner_x - corner_size, corner_y), 
                                   (corner_x + corner_size, corner_y), width=1)
            self._draw_sketchy_line(draw, (corner_x, corner_y - corner_size), 
                                   (corner_x, corner_y + corner_size), width=1)
    
    def _apply_sketchy_effects(self, map_image: np.ndarray) -> np.ndarray:
        """Apply hand-drawn, sketchy effects to the entire map."""
        # Add slight texture to simulate paper
        texture_noise = np.random.random(map_image.shape) * 10 - 5
        map_image = np.clip(map_image.astype(float) + texture_noise, 0, 255).astype(np.uint8)
        
        # Slight blur for hand-drawn feel
        map_image = cv2.GaussianBlur(map_image, (1, 1), 0)
        
        return map_image
    
    # Utility drawing methods
    def _draw_sketchy_line(self, draw: ImageDraw.Draw, start: Tuple[float, float], 
                          end: Tuple[float, float], width: int = 1, color: Optional[Tuple[int, int, int]] = None):
        """Draw a line with hand-drawn, sketchy appearance."""
        if color is None:
            color = (0, 0, 0)  # Black
        
        # Add multiple slightly offset lines for sketchy effect
        for offset in range(-width//2, width//2 + 1):
            for _ in range(2):  # Draw each line twice with slight variation
                start_x = start[0] + np.random.uniform(-0.5, 0.5)
                start_y = start[1] + np.random.uniform(-0.5, 0.5) + offset
                end_x = end[0] + np.random.uniform(-0.5, 0.5)
                end_y = end[1] + np.random.uniform(-0.5, 0.5) + offset
                
                try:
                    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=1)
                except:
                    pass  # Skip invalid coordinates
    
    def _draw_sketchy_circle(self, draw: ImageDraw.Draw, x: int, y: int, radius: int, 
                            color: Optional[Tuple[int, int, int]] = None):
        """Draw a circle with sketchy appearance."""
        if color is None:
            color = (0, 0, 0)
        
        # Draw circle as multiple arc segments with slight variations
        num_segments = 16
        angle_step = 2 * np.pi / num_segments
        
        points = []
        for i in range(num_segments + 1):
            angle = i * angle_step
            # Add randomness to radius for sketchy effect
            r = radius + np.random.uniform(-1, 1)
            px = x + r * np.cos(angle)
            py = y + r * np.sin(angle)
            points.append((px, py))
        
        # Draw connecting lines
        for i in range(len(points) - 1):
            self._draw_sketchy_line(draw, points[i], points[i+1], width=1, color=color)
    
    def _group_nearby_points(self, points: Tuple[np.ndarray, np.ndarray], 
                           distance_threshold: float = 30) -> List[List[Tuple[int, int]]]:
        """Group nearby points together."""
        coordinates = list(zip(points[0], points[1]))
        if not coordinates:
            return []
        
        groups = []
        remaining = set(range(len(coordinates)))
        
        while remaining:
            current_group = []
            queue = [remaining.pop()]
            
            while queue:
                current_idx = queue.pop(0)
                current_group.append(coordinates[current_idx])
                
                # Find nearby points
                current_point = coordinates[current_idx]
                to_remove = []
                
                for other_idx in remaining:
                    other_point = coordinates[other_idx]
                    distance = np.sqrt((current_point[0] - other_point[0])**2 + 
                                     (current_point[1] - other_point[1])**2)
                    
                    if distance <= distance_threshold:
                        queue.append(other_idx)
                        to_remove.append(other_idx)
                
                for idx in to_remove:
                    remaining.remove(idx)
            
            if len(current_group) > 5:  # Only keep substantial groups
                groups.append(current_group)
        
        return groups


# Integration function for use with existing cartography system
def create_fantasy_terrain_overlay(base_mask: np.ndarray, width: int, height: int, 
                                 seed: Optional[int] = None) -> np.ndarray:
    """
    Create a fantasy terrain overlay for an existing map.
    This function can be called from the main cartography system.
    """
    artist = FantasyTerrainArtist(width, height, seed)
    return artist.create_fantasy_map(base_mask)


if __name__ == "__main__":
    # Example usage
    width, height = 800, 600
    
    # Create a simple test mask (you would use your actual map mask)
    test_mask = np.zeros((height, width), dtype=np.uint8)
    # Create some land areas for testing
    cv2.circle(test_mask, (400, 300), 200, 255, -1)
    cv2.circle(test_mask, (600, 200), 100, 255, -1)
    
    # Generate fantasy terrain
    fantasy_map = create_fantasy_terrain_overlay(test_mask, width, height, seed=42)
    
    print(f"Generated fantasy map with shape {fantasy_map.shape}")
    # Optional: save result
    cv2.imwrite('fantasy_map_test.png', cv2.cvtColor(fantasy_map, cv2.COLOR_RGB2BGR))
