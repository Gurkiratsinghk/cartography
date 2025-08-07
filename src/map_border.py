import numpy as np
import noise
import cv2

def generate_border(width, height, scale=100.0, octaves=6, persistence=0.5, lacunarity=2.0):
    """Generates a natural-looking map border using Perlin noise."""
    border = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            border[i][j] = noise.pnoise2(i/scale,
                                         j/scale,
                                         octaves=octaves,
                                         persistence=persistence,
                                         lacunarity=lacunarity,
                                         repeatx=width,
                                         repeaty=height,
                                         base=0)

    # Normalize the border to be between 0 and 255
    border = ((border - border.min()) * (1/(border.max() - border.min()) * 255)).astype(np.uint8)

    # Threshold the border to create a binary mask
    _, border_mask = cv2.threshold(border, 127, 255, cv2.THRESH_BINARY)

    return border_mask
