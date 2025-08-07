import cv2
import numpy as np
from PIL import Image

def binarize_image(image, invert=False):
    """Converts an image to a black and white image."""
    cv_image = np.array(image.convert('RGB'))
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
    binary_image = cv2.adaptiveThreshold(gray_image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    if invert:
        binary_image = cv2.bitwise_not(binary_image)
    return Image.fromarray(binary_image)

def generate_textures(image):
    """Generates textures for the binarized image."""
    color_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2RGB)
    land_color = [0, 255, 0]  # Green
    water_color = [255, 0, 0]  # Blue
    land_mask = np.all(color_image == [255, 255, 255], axis=-1)
    color_image[land_mask] = land_color
    color_image[~land_mask] = water_color
    return Image.fromarray(color_image)

def apply_rules(image):
    """Applies rules to the textured image."""
    image_array = np.array(image)
    land_color = [0, 255, 0]  # Green
    water_color = [255, 0, 0]  # Blue
    beach_color = [0, 255, 255]  # Yellow
    new_image_array = image_array.copy()
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            if np.array_equal(image_array[y, x], land_color):
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
    return Image.fromarray(new_image_array)
