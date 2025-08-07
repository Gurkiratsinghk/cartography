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
        moisture_from_water = 1 - (distance_from_water / max_dist)
        
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
        
        if len(mountain_points[0]) == 0:
            return map_image
        
        # Convert to PIL for drawing
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Group mountain points into ranges
        mountain_clusters = self._cluster_points(mountain_points, cluster_distance=30)
        
        for cluster in mountain_clusters:
            # Draw mountain symbols for each cluster
            for y, x in cluster[::5]:  # Sample every 5th point to avoid overcrowding
                self._draw_mountain_symbol(draw, x, y, size=np.random.randint(8, 15))
        
        return np.array(pil_image)
    
    def _add_forest_areas(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Add hand-drawn forest symbols."""
        forest_mask = terrain_map == 2
        forest_points = np.where(forest_mask)
        
        if len(forest_points[0]) == 0:
            return map_image
        
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Sample forest points to avoid overcrowding
        num_trees = min(len(forest_points[0]) // 10, 200)
        if num_trees > 0:
            indices = np.random.choice(len(forest_points[0]), num_trees, replace=False)
            
            for idx in indices:
                y, x = forest_points[0][idx], forest_points[1][idx]
                tree_size = np.random.randint(4, 8)
                self._draw_tree_symbol(draw, x, y, size=tree_size)
        
        return np.array(pil_image)
    
    def _add_hills_and_elevation(self, map_image: np.ndarray, terrain_map: np.ndarray) -> np.ndarray:
        """Add elevation contours and hill shading."""
        hills_mask = terrain_map == 3
        
        if not np.any(hills_mask):
            return map_image
        
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Find hill areas and add contour-like lines
        contours, _ = cv2.findContours(hills_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 10:
                # Draw irregular contour lines
                self._draw_hill_contours(draw, contour)
        
        return np.array(pil_image)
    
    def _add_rivers_and_streams(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add winding rivers from mountains to sea."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Find potential river sources (high elevation areas near land)
        land_mask = base_mask == 255
        
        # Generate a few major rivers
        num_rivers = np.random.randint(2, 5)
        
        for _ in range(num_rivers):
            # Find a starting point inland
            land_points = np.where(land_mask)
            if len(land_points[0]) > 0:
                start_idx = np.random.randint(0, len(land_points[0]))
                start_y, start_x = land_points[0][start_idx], land_points[1][start_idx]
                
                river_path = self._generate_river_path(start_x, start_y, base_mask)
                self._draw_river(draw, river_path)
        
        return np.array(pil_image)
    
    def _add_roads_and_paths(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add roads connecting settlements."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Generate some road networks
        land_mask = base_mask == 255
        land_points = np.where(land_mask)
        
        if len(land_points[0]) > 0:
            # Create a few road segments
            num_roads = np.random.randint(3, 8)
            
            for _ in range(num_roads):
                # Random start and end points on land
                start_idx = np.random.randint(0, len(land_points[0]))
                end_idx = np.random.randint(0, len(land_points[0]))
                
                start_y, start_x = land_points[0][start_idx], land_points[1][start_idx]
                end_y, end_x = land_points[0][end_idx], land_points[1][end_idx]
                
                # Only draw road if points are reasonably far apart
                distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                if distance > 50:
                    road_path = self._generate_road_path(start_x, start_y, end_x, end_y, base_mask)
                    self._draw_road(draw, road_path)
        
        return np.array(pil_image)
    
    def _add_settlements(self, map_image: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Add settlements like towns and cities."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        land_mask = base_mask == 255
        land_points = np.where(land_mask)
        
        if len(land_points[0]) > 0:
            # Add various types of settlements
            num_cities = np.random.randint(1, 3)
            num_towns = np.random.randint(2, 6)
            num_villages = np.random.randint(4, 10)
            
            # Cities (large settlements)
            for _ in range(num_cities):
                idx = np.random.randint(0, len(land_points[0]))
                y, x = land_points[0][idx], land_points[1][idx]
                self._draw_city(draw, x, y)
            
            # Towns (medium settlements)
            for _ in range(num_towns):
                idx = np.random.randint(0, len(land_points[0]))
                y, x = land_points[0][idx], land_points[1][idx]
                self._draw_town(draw, x, y)
            
            # Villages (small settlements)
            for _ in range(num_villages):
                idx = np.random.randint(0, len(land_points[0]))
                y, x = land_points[0][idx], land_points[1][idx]
                self._draw_village(draw, x, y)
        
        return np.array(pil_image)
    
    def _add_decorative_elements(self, map_image: np.ndarray) -> np.ndarray:
        """Add decorative elements like compass roses, sea monsters, etc."""
        pil_image = Image.fromarray(map_image)
        draw = ImageDraw.Draw(pil_image)
        
        # Add compass rose
        compass_x = self.width - 80
        compass_y = 80
        self._draw_compass_rose(draw, compass_x, compass_y)
        
        # Add some sea creatures in water areas
        water_mask = np.array(pil_image)
        water_points = np.where(np.all(water_mask == self.terrain_colors['water'], axis=2))
        
        if len(water_points[0]) > 0:
            num_creatures = np.random.randint(1, 4)
            for _ in range(num_creatures):
                idx = np.random.randint(0, len(water_points[0]))
                y, x = water_points[0][idx], water_points[1][idx]
                self._draw_sea_creature(draw, x, y)
        
        return np.array(pil_image)
    
    def _apply_sketchy_effects(self, map_image: np.ndarray) -> np.ndarray:
        """Apply effects to make the map look hand-drawn."""
        # Add slight paper texture
        map_image = self._add_paper_texture(map_image)
        
        # Add subtle aging effects
        map_image = self._add_aging_effects(map_image)
        
        return map_image
    
    # Helper drawing methods
    def _draw_mountain_symbol(self, draw: ImageDraw.Draw, x: int, y: int, size: int = 10):
        """Draw a single mountain symbol."""
        # Draw triangular mountain shape with some randomness
        height = size
        width = size * 0.8
        
        # Main triangle
        points = [
            (x - width/2 + np.random.randint(-2, 3), y + height/2),
            (x + np.random.randint(-2, 3), y - height/2 + np.random.randint(-1, 2)),
            (x + width/2 + np.random.randint(-2, 3), y + height/2)
        ]
        
        draw.polygon(points, outline=(0, 0, 0), width=1)
        
        # Add some shading lines
        for i in range(2, size//2):
            start_x = x - width/4 + i
            start_y = y - height/4 + i//2
            end_x = x + width/4 - i
            end_y = y + height/4 - i//2
            if 0 <= start_x < self.width and 0 <= start_y < self.height:
                draw.line([(start_x, start_y), (end_x, end_y)], fill=(100, 100, 100), width=1)
    
    def _draw_tree_symbol(self, draw: ImageDraw.Draw, x: int, y: int, size: int = 6):
        """Draw a simple tree symbol."""
        # Tree trunk
        trunk_height = size // 3
        draw.line([(x, y), (x, y + trunk_height)], fill=(139, 69, 19), width=2)
        
        # Tree crown (circle with irregular edge)
        crown_radius = size // 2
        crown_points = []
        num_points = 8
        
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            radius_variation = crown_radius + np.random.randint(-2, 3)
            point_x = x + radius_variation * np.cos(angle)
            point_y = y - crown_radius + radius_variation * np.sin(angle)
            crown_points.append((point_x, point_y))
        
        draw.polygon(crown_points, outline=(0, 100, 0), fill=self.terrain_colors['forest'], width=1)
    
    def _draw_hill_contours(self, draw: ImageDraw.Draw, contour: np.ndarray):
        """Draw elevation contour lines for hills."""
        # Smooth the contour and draw multiple lines
        if len(contour) < 4:
            return
            
        points = contour.reshape(-1, 2)
        
        # Draw multiple concentric contour lines
        for offset in range(0, 10, 3):
            offset_points = []
            for point in points[::2]:  # Sample every other point
                x, y = point
                # Add slight random offset for hand-drawn look
                x += np.random.randint(-1, 2) + offset//3
                y += np.random.randint(-1, 2) + offset//3
                offset_points.append((x, y))
            
            if len(offset_points) > 2:
                draw.line(offset_points, fill=(139, 137, 137), width=1)
    
    def _generate_river_path(self, start_x: int, start_y: int, base_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Generate a winding river path toward the sea."""
        path = [(start_x, start_y)]
        current_x, current_y = start_x, start_y
        
        # Try to flow toward water
        water_points = np.where(base_mask == 0)
        if len(water_points[0]) == 0:
            return path
        
        # Find nearest water
        distances = np.sqrt((water_points[1] - current_x)**2 + (water_points[0] - current_y)**2)
        nearest_water_idx = np.argmin(distances)
        target_x, target_y = water_points[1][nearest_water_idx], water_points[0][nearest_water_idx]
        
        # Generate meandering path
        steps = max(int(np.sqrt((target_x - start_x)**2 + (target_y - start_y)**2)), 10)
        
        for i in range(steps):
            # General direction toward target
            progress = i / steps
            direct_x = start_x + progress * (target_x - start_x)
            direct_y = start_y + progress * (target_y - start_y)
            
            # Add meandering
            meander_strength = 20
            meander_x = direct_x + meander_strength * np.sin(i * 0.3) * np.random.uniform(0.5, 1.5)
            meander_y = direct_y + meander_strength * np.cos(i * 0.2) * np.random.uniform(0.5, 1.5)
            
            current_x = int(np.clip(meander_x, 0, self.width - 1))
            current_y = int(np.clip(meander_y, 0, self.height - 1))
            path.append((current_x, current_y))
        
        return path
    
    def _generate_road_path(self, start_x: int, start_y: int, end_x: int, end_y: int, base_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Generate a road path between two points."""
        path = [(start_x, start_y)]
        
        steps = max(int(np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2) / 10), 5)
        
        for i in range(1, steps):
            progress = i / steps
            # Linear interpolation with slight randomness
            x = int(start_x + progress * (end_x - start_x) + np.random.randint(-5, 6))
            y = int(start_y + progress * (end_y - start_y) + np.random.randint(-5, 6))
            
            # Keep on land if possible
            x = np.clip(x, 0, self.width - 1)
            y = np.clip(y, 0, self.height - 1)
            
            path.append((x, y))
        
        path.append((end_x, end_y))
        return path
    
    def _draw_river(self, draw: ImageDraw.Draw, path: List[Tuple[int, int]]):
        """Draw a river along the given path."""
        if len(path) < 2:
            return
        
        # Draw river with varying width
        for i in range(len(path) - 1):
            start_point = path[i]
            end_point = path[i + 1]
            
            # Rivers get wider as they approach the sea
            width = 2 + i // 10
            draw.line([start_point, end_point], fill=self.terrain_colors['water'], width=width)
    
    def _draw_road(self, draw: ImageDraw.Draw, path: List[Tuple[int, int]]):
        """Draw a road along the given path."""
        if len(path) < 2:
            return
        
        # Draw road as dashed line
        for i in range(0, len(path) - 1, 3):  # Dashed effect
            if i + 1 < len(path):
                start_point = path[i]
                end_point = path[i + 1]
                draw.line([start_point, end_point], fill=self.terrain_colors['roads'], width=1)
    
    def _draw_city(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a city symbol."""
        # Large square with towers
        size = 12
        draw.rectangle([x - size//2, y - size//2, x + size//2, y + size//2], 
                      outline=(0, 0, 0), width=2)
        
        # Add towers
        for dx, dy in [(-size//3, -size//2), (size//3, -size//2)]:
            tower_x, tower_y = x + dx, y + dy
            draw.rectangle([tower_x - 2, tower_y - 4, tower_x + 2, tower_y], 
                          fill=(0, 0, 0))
    
    def _draw_town(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a town symbol."""
        # Medium circle
        size = 8
        draw.ellipse([x - size//2, y - size//2, x + size//2, y + size//2], 
                    outline=(0, 0, 0), width=2)
        
        # Cross in center
        draw.line([(x - size//4, y), (x + size//4, y)], fill=(0, 0, 0), width=1)
        draw.line([(x, y - size//4), (x, y + size//4)], fill=(0, 0, 0), width=1)
    
    def _draw_village(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a village symbol."""
        # Small square
        size = 4
        draw.rectangle([x - size//2, y - size//2, x + size//2, y + size//2], 
                      outline=(0, 0, 0), width=1)
    
    def _draw_compass_rose(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a decorative compass rose."""
        size = 30
        
        # Main directions
        directions = ['N', 'E', 'S', 'W']
        angles = [0, 90, 180, 270]
        
        for direction, angle in zip(directions, angles):
            angle_rad = np.radians(angle)
            end_x = x + size * np.cos(angle_rad)
            end_y = y + size * np.sin(angle_rad)
            
            # Draw direction line
            draw.line([(x, y), (end_x, end_y)], fill=(0, 0, 0), width=2)
            
            # Draw direction label
            text_x = x + (size + 10) * np.cos(angle_rad) - 5
            text_y = y + (size + 10) * np.sin(angle_rad) - 5
            draw.text((text_x, text_y), direction, fill=(0, 0, 0))
        
        # Center circle
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 0, 0))
    
    def _draw_sea_creature(self, draw: ImageDraw.Draw, x: int, y: int):
        """Draw a decorative sea creature."""
        # Simple sea serpent
        body_length = 20
        segments = 5
        
        for i in range(segments):
            segment_x = x + i * (body_length // segments)
            segment_y = y + 5 * np.sin(i * 0.5)
            
            if i == 0:  # Head
                draw.ellipse([segment_x - 3, segment_y - 3, segment_x + 3, segment_y + 3], 
                           fill=(0, 100, 0))
            else:  # Body
                draw.ellipse([segment_x - 2, segment_y - 2, segment_x + 2, segment_y + 2], 
                           fill=(0, 150, 0))
    
    def _cluster_points(self, points: Tuple[np.ndarray, np.ndarray], cluster_distance: int = 20) -> List[List[Tuple[int, int]]]:
        """Group nearby points into clusters."""
        if len(points[0]) == 0:
            return []
        
        # Convert to list of coordinate pairs
        coords = list(zip(points[0], points[1]))
        
        clusters = []
        unassigned = coords.copy()
        
        while unassigned:
            # Start new cluster with first unassigned point
            current_cluster = [unassigned.pop(0)]
            
            # Find all points within cluster distance
            i = 0
            while i < len(unassigned):
                point = unassigned[i]
                
                # Check distance to any point in current cluster
                min_distance = min(
                    np.sqrt((point[0] - cluster_point[0])**2 + (point[1] - cluster_point[1])**2)
                    for cluster_point in current_cluster
                )
                
                if min_distance <= cluster_distance:
                    current_cluster.append(unassigned.pop(i))
                else:
                    i += 1
            
            clusters.append(current_cluster)
        
        return clusters
    
    def _add_paper_texture(self, map_image: np.ndarray) -> np.ndarray:
        """Add subtle paper texture to the map."""
        # Generate subtle noise
        noise = np.random.normal(0, 5, map_image.shape).astype(np.int16)
        
        # Apply noise
        textured = map_image.astype(np.int16) + noise
        textured = np.clip(textured, 0, 255).astype(np.uint8)
        
        return textured
    
    def _add_aging_effects(self, map_image: np.ndarray) -> np.ndarray:
        """Add subtle aging effects like slight yellowing."""
        # Slight sepia tone
        aged = map_image.copy().astype(np.float32)
        
        # Sepia transformation matrix
        sepia_filter = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply very subtle sepia (mix with original)
        sepia_aged = aged.dot(sepia_filter.T)
        
        # Blend original with sepia (90% original, 10% sepia)
        result = 0.95 * aged + 0.05 * sepia_aged
        
        return np.clip(result, 0, 255).astype(np.uint8)


def create_enhanced_fantasy_map(base_mask: np.ndarray, width: int, height: int, seed: int = 42) -> np.ndarray:
    """
    Main function to create a complete fantasy map with enhanced terrain features.
    
    Args:
        base_mask: Binary mask where 255 = land, 0 = water
        width, height: Dimensions of the map
        seed: Random seed for reproducible results
    
    Returns:
        RGB image array of the fantasy map
    """
    artist = FantasyTerrainArtist(width, height, seed)
    fantasy_map = artist.create_fantasy_map(base_mask, terrain_density=0.7)
    return fantasy_map


if __name__ == "__main__":
    # Example usage
    width, height = 800, 600
    
    # Create a simple test mask (circular island)
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 3
    
    y, x = np.ogrid[:height, :width]
    test_mask = ((x - center_x)**2 + (y - center_y)**2 <= radius**2).astype(np.uint8) * 255
    
    # Generate fantasy map
    fantasy_map = create_enhanced_fantasy_map(test_mask, width, height, seed=42)
    
    print(f"Generated fantasy map with shape: {fantasy_map.shape}")
    
    # Optional: Save the result
    # from PIL import Image
    # Image.fromarray(fantasy_map).save('fantasy_map_test.png')