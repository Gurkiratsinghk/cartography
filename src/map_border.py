import numpy as np
from perlin_noise import PerlinNoise
import cv2

def generate_border(width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """Generates a natural-looking map border using Perlin noise."""
    perlin = PerlinNoise(octaves=octaves, seed=0)
    border = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            border[i][j] = perlin([i/scale, j/scale])

    # Normalize the border to be between 0 and 255
    border = ((border - border.min()) * (1/(border.max() - border.min()) * 255)).astype(np.uint8)

    # Threshold the border to create a binary mask
    _, border_mask = cv2.threshold(border, 127, 255, cv2.THRESH_BINARY)

    return border_mask
