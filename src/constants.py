import numpy as np
from pygame.math import Vector2 as v

# COLORS
BLACK = 0, 0, 0
RED = 255, 0, 0
GREEN = 0, 255, 0
BLUE = 0, 0, 255
GREY = 128, 128, 128
WHITE = 255, 255, 255

# Link lengths
L1 = 80
L2 = 70

# Robot Name
NAME = "2 DoF Planar Robot"

# Working Range
WORKING_RANGE = np.array([[-np.pi, np.pi], [-3*np.pi/4, 3*np.pi/4]])

# Screen Size
WIDTH, HEIGHT = 640, 480

# Screen Center
START = v(WIDTH // 2, HEIGHT // 2)

def _to_zero(vector) -> v:
    # make rigid transformation to move the vector to the center of the screen
    return vector + START