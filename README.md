# Cartography

Cartography is a Python program for generating maps with two primary modes:

- **World Map Mode:** Generate a full world map with a defined topography. All subsequent maps derived from this world map will adhere to its topographic structure, ensuring consistency. The program supports creating multiple distinct world maps.
- **Simple Map Mode:** Generate a specific, smaller part of a larger world map, focusing on a particular region while maintaining consistency with the parent world map's topography.

## Features

- Load an image to use as a base for the map.
- Binarize the image to create a black and white map, representing land and water.
- Invert the binarization to choose whether land or water is black.
- Overlay a square or hexagonal grid on the map.
- Adjust the grid size with a slider.

## How to Run

1.  Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```
2.  Run the application:
    ```
    python main.py
    ```

## To-Do

- Give texture and rules.
- Implement "Simple Map Mode" for selecting a region of the map.
