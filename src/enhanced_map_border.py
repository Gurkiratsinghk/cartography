import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import random
from typing import Tuple, List, Optional

class EnhancedMapBorder:
    """Enhanced map border generator with multiple algorithms for creating natural-looking landmasses."""
    
    def __init__(self, width: int, height: int, seed: Optional[int] = None):
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_island_chain(self, num_islands: int = 3, size_variation: float = 0.3) -> np.ndarray:
        """Generate a chain of connected islands with natural variation."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create main island positions along a curved path
        t = np.linspace(0, 2 * np.pi, num_islands + 1)[:-1]
        curve_radius = min(self.width, self.height) * 0.2
        curve_center_offset = 0.1
        
        for i, angle in enumerate(t):
            # Position islands along a curved path
            island_x = self.center_x + curve_radius * np.cos(angle) + \
                      np.random.normal(0, self.width * curve_center_offset)
            island_y = self.center_y + curve_radius * np.sin(angle) + \
                      np.random.normal(0, self.height * curve_center_offset)
            
            # Vary island sizes
            base_size = min(self.width, self.height) * 0.15
            island_size = base_size * (1 + np.random.uniform(-size_variation, size_variation))
            
            # Create irregular island shape
            island_mask = self._create_irregular_island(
                int(island_x), int(island_y), int(island_size)
            )
            mask = cv2.bitwise_or(mask, island_mask)
        
        # Connect islands with land bridges
        mask = self._add_land_bridges(mask)
        
        # Smooth and add coastal detail
        mask = self._add_coastal_detail(mask)
        
        return mask
    
    def generate_continent(self, fractal_dimension: float = 1.5, roughness: float = 0.7) -> np.ndarray:
        """Generate a large continental landmass with fractal coastline."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Start with elliptical base
        center_x, center_y = self.center_x, self.center_y
        a = self.width * 0.3  # Semi-major axis
        b = self.height * 0.25  # Semi-minor axis
        
        # Create base ellipse
        y, x = np.ogrid[:self.height, :self.width]
        ellipse_mask = ((x - center_x)**2 / a**2 + (y - center_y)**2 / b**2) <= 1
        
        # Apply fractal distortion to coastline
        noise = self._generate_fractal_noise(fractal_dimension, roughness)
        
        # Distort the coastline using the noise
        distorted_mask = self._apply_coastline_distortion(ellipse_mask, noise)
        
        # Add peninsulas and bays
        distorted_mask = self._add_peninsulas_and_bays(distorted_mask)
        
        # Final smoothing and detail
        mask = self._add_coastal_detail(distorted_mask.astype(np.uint8) * 255)
        
        return mask
    
    def generate_archipelago(self, density: float = 0.3, size_range: Tuple[int, int] = (20, 80)) -> np.ndarray:
        """Generate an archipelago with many small to medium islands."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Calculate number of islands based on density
        area = self.width * self.height
        num_islands = int(area * density / 10000)  # Roughly based on area
        
        # Generate islands with clustering
        clusters = self._generate_island_clusters(num_islands, size_range)
        
        for cluster in clusters:
            for island_x, island_y, island_size in cluster:
                if 0 <= island_x < self.width and 0 <= island_y < self.height:
                    island_mask = self._create_irregular_island(island_x, island_y, island_size)
                    mask = cv2.bitwise_or(mask, island_mask)
        
        # Add some connecting reefs and sandbars
        mask = self._add_reefs_and_sandbars(mask)
        
        return mask
    
    def generate_fjord_coastline(self, num_fjords: int = 5, depth_factor: float = 0.8) -> np.ndarray:
        """Generate a coastline with dramatic fjords and inlets."""
        # Start with base landmass
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Create base coastal shape
        base_radius = min(self.width, self.height) * 0.35
        y, x = np.ogrid[:self.height, :self.width]
        base_mask = ((x - self.center_x)**2 + (y - self.center_y)**2) <= base_radius**2
        
        # Carve out fjords
        for _ in range(num_fjords):
            fjord_mask = self._create_fjord(base_radius, depth_factor)
            base_mask = base_mask & (~fjord_mask)
        
        # Convert to uint8 and add detail
        mask = base_mask.astype(np.uint8) * 255
        mask = self._add_coastal_detail(mask)
        
        return mask
    
    def _create_irregular_island(self, center_x: int, center_y: int, size: int) -> np.ndarray:
        """Create an irregularly shaped island using multiple methods."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Number of control points for shape
        num_points = np.random.randint(8, 16)
        angles = np.linspace(0, 2 * np.pi, num_points + 1)[:-1]
        
        # Generate irregular radius for each angle
        base_radius = size
        radii = []
        for i, angle in enumerate(angles):
            # Add noise to radius
            noise_factor = np.random.uniform(0.5, 1.5)
            radius = base_radius * noise_factor
            radii.append(radius)
        
        # Create polygon points
        points = []
        for angle, radius in zip(angles, radii):
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            points.append([int(x), int(y)])
        
        # Fill polygon
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        
        # Smooth the edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _generate_fractal_noise(self, fractal_dim: float, roughness: float) -> np.ndarray:
        """Generate fractal noise for coastline distortion."""
        noise = np.zeros((self.height, self.width))
        
        # Multiple octaves of noise
        frequency = 1.0
        amplitude = 1.0
        
        for _ in range(6):  # 6 octaves
            octave_noise = np.random.rand(self.height, self.width) * 2 - 1
            octave_noise = gaussian_filter(octave_noise, sigma=10.0/frequency)
            noise += amplitude * octave_noise
            
            frequency *= 2.0
            amplitude *= roughness
        
        return noise
    
    def _apply_coastline_distortion(self, base_mask: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """Apply fractal distortion to coastline."""
        # Normalize noise
        noise = (noise - noise.min()) / (noise.max() - noise.min()) * 2 - 1
        
        # Create distance transform from coastline
        distance_transform = cv2.distanceTransform(
            base_mask.astype(np.uint8), cv2.DIST_L2, 5
        )
        
        # Apply distortion based on distance from coast
        distortion_strength = 20.0  # pixels
        threshold_adjustment = noise * distortion_strength
        
        # Adjust the distance threshold based on noise
        distorted_mask = distance_transform > threshold_adjustment
        
        return distorted_mask
    
    def _add_peninsulas_and_bays(self, mask: np.ndarray) -> np.ndarray:
        """Add peninsulas extending out and bays cutting in."""
        # Find coastline points
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return mask
        
        # Work with the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Add peninsulas (extend outward from random points)
        num_peninsulas = np.random.randint(2, 6)
        for _ in range(num_peninsulas):
            if len(largest_contour) > 0:
                # Random point on coastline
                idx = np.random.randint(0, len(largest_contour))
                point = largest_contour[idx][0]
                
                # Create peninsula extending outward
                peninsula = self._create_peninsula(point, mask)
                mask = mask | peninsula
        
        return mask
    
    def _create_peninsula(self, start_point: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
        """Create a peninsula extending from a coastline point."""
        peninsula_mask = np.zeros_like(base_mask, dtype=bool)
        
        x, y = start_point
        
        # Random direction for peninsula (roughly outward from land)
        angles = np.linspace(0, 2*np.pi, 8)
        best_angle = 0
        max_water_distance = 0
        
        # Find direction with most water (outward direction)
        for angle in angles:
            test_x = x + 50 * np.cos(angle)
            test_y = y + 50 * np.sin(angle)
            if (0 <= test_x < self.width and 0 <= test_y < self.height and 
                not base_mask[int(test_y), int(test_x)]):
                distance = np.sqrt((test_x - x)**2 + (test_y - y)**2)
                if distance > max_water_distance:
                    max_water_distance = distance
                    best_angle = angle
        
        # Create peninsula along best direction
        peninsula_length = np.random.randint(30, 80)
        peninsula_width = np.random.randint(10, 25)
        
        for i in range(peninsula_length):
            # Add some wiggle to the peninsula
            angle_variation = np.random.normal(0, 0.1)
            current_angle = best_angle + angle_variation
            
            center_x = x + i * np.cos(current_angle)
            center_y = y + i * np.sin(current_angle)
            
            # Taper the width as we go further out
            current_width = peninsula_width * (1 - i / peninsula_length)
            
            # Draw ellipse for this segment
            if 0 <= center_x < self.width and 0 <= center_y < self.height:
                rr, cc = np.ogrid[:self.height, :self.width]
                mask_segment = ((rr - center_y)**2 / current_width**2 + 
                              (cc - center_x)**2 / (current_width/2)**2) <= 1
                peninsula_mask |= mask_segment
        
        return peninsula_mask
    
    def _generate_island_clusters(self, num_islands: int, size_range: Tuple[int, int]) -> List[List[Tuple[int, int, int]]]:
        """Generate clusters of islands with natural spacing."""
        clusters = []
        num_clusters = max(1, num_islands // 5)  # Group islands into clusters
        
        for _ in range(num_clusters):
            # Random cluster center
            cluster_x = np.random.randint(size_range[1], self.width - size_range[1])
            cluster_y = np.random.randint(size_range[1], self.height - size_range[1])
            
            # Islands in this cluster
            cluster_size = np.random.randint(3, 8)
            cluster_islands = []
            
            for _ in range(cluster_size):
                # Position relative to cluster center
                offset_x = np.random.normal(0, 60)
                offset_y = np.random.normal(0, 60)
                
                island_x = cluster_x + offset_x
                island_y = cluster_y + offset_y
                island_size = np.random.randint(size_range[0], size_range[1])
                
                cluster_islands.append((int(island_x), int(island_y), island_size))
            
            clusters.append(cluster_islands)
        
        return clusters
    
    def _add_land_bridges(self, mask: np.ndarray) -> np.ndarray:
        """Add narrow land connections between nearby islands."""
        # Find island contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return mask
        
        # Connect nearby islands
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                # Find closest points between contours
                pts1 = contours[i].reshape(-1, 2)
                pts2 = contours[j].reshape(-1, 2)
                
                distances = cdist(pts1, pts2)
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                min_distance = distances[min_idx]
                
                # Connect if close enough
                if min_distance < 100:  # pixels
                    pt1 = pts1[min_idx[0]]
                    pt2 = pts2[min_idx[1]]
                    
                    # Draw connection with some width variation
                    width = np.random.randint(3, 8)
                    cv2.line(mask, tuple(pt1), tuple(pt2), 255, width)
        
        return mask
    
    def _create_fjord(self, base_radius: float, depth_factor: float) -> np.ndarray:
        """Create a single fjord cutting into the coastline."""
        fjord_mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Random angle for fjord direction
        angle = np.random.uniform(0, 2 * np.pi)
        
        # Start point on the perimeter
        start_x = self.center_x + base_radius * 0.8 * np.cos(angle)
        start_y = self.center_y + base_radius * 0.8 * np.sin(angle)
        
        # Fjord parameters
        fjord_length = int(base_radius * depth_factor)
        max_width = np.random.randint(15, 30)
        
        # Create fjord path with meandering
        current_x, current_y = start_x, start_y
        current_angle = angle + np.pi  # Point inward
        
        for i in range(fjord_length):
            # Add meandering
            angle_change = np.random.normal(0, 0.05)
            current_angle += angle_change
            
            # Move inward
            current_x += np.cos(current_angle)
            current_y += np.sin(current_angle)
            
            # Taper width as we go deeper
            width = max_width * (1 - i / fjord_length) * 0.5
            
            # Draw fjord segment
            if 0 <= current_x < self.width and 0 <= current_y < self.height:
                rr, cc = np.ogrid[:self.height, :self.width]
                segment = ((rr - current_y)**2 + (cc - current_x)**2) <= width**2
                fjord_mask |= segment
        
        return fjord_mask
    
    def _add_reefs_and_sandbars(self, mask: np.ndarray) -> np.ndarray:
        """Add small reefs and sandbars around islands."""
        # Find water areas near islands
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        reef_area = dilated & (~mask)  # Areas near land but in water
        
        # Add small reef patches
        num_reefs = np.random.randint(5, 15)
        reef_locations = np.where(reef_area == 255)
        
        if len(reef_locations[0]) > 0:
            for _ in range(min(num_reefs, len(reef_locations[0]))):
                idx = np.random.randint(0, len(reef_locations[0]))
                reef_y, reef_x = reef_locations[0][idx], reef_locations[1][idx]
                
                # Small reef patch
                reef_size = np.random.randint(3, 8)
                rr, cc = np.ogrid[:self.height, :self.width]
                reef_patch = ((rr - reef_y)**2 + (cc - reef_x)**2) <= reef_size**2
                mask[reef_patch] = 255
        
        return mask
    
    def _add_coastal_detail(self, mask: np.ndarray) -> np.ndarray:
        """Add fine coastal detail and smooth the coastline."""
        # Light smoothing to remove pixelation
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Add some small coastal features
        mask = self._add_small_bays_and_capes(mask)
        
        return mask
    
    def _add_small_bays_and_capes(self, mask: np.ndarray) -> np.ndarray:
        """Add small bays and capes for coastal detail."""
        # Find coastline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) > 20:  # Only process substantial coastlines
                # Add random small indentations and protrusions
                num_features = len(contour) // 10
                
                for _ in range(num_features):
                    idx = np.random.randint(0, len(contour))
                    point = contour[idx][0]
                    
                    feature_size = np.random.randint(3, 8)
                    if np.random.random() > 0.5:
                        # Add small bay (erosion)
                        cv2.circle(mask, tuple(point), feature_size, 0, -1)
                    else:
                        # Add small cape (dilation)
                        cv2.circle(mask, tuple(point), feature_size, 255, -1)
        
        return mask


def test_enhanced_borders():
    """Test function to generate sample borders using different algorithms."""
    width, height = 800, 600
    generator = EnhancedMapBorder(width, height, seed=42)
    
    # Test different border types
    border_types = {
        'island_chain': generator.generate_island_chain(num_islands=4),
        'continent': generator.generate_continent(roughness=0.8),
        'archipelago': generator.generate_archipelago(density=0.4),
        'fjord_coastline': generator.generate_fjord_coastline(num_fjords=6)
    }
    
    return border_types

if __name__ == "__main__":
    # Example usage
    borders = test_enhanced_borders()
    
    # You could save these or display them
    for name, border in borders.items():
        print(f"Generated {name} border with shape {border.shape}")
        # Optional: save to file
        # cv2.imwrite(f'{name}_border.png', border)