from re import M
import numpy as np
from pygame.math import Vector2 as v
from pygame import Surface, draw, transform

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

# Working Range (in radians)
WORKING_RANGE = np.array([[-np.pi, np.pi], [-3 * np.pi / 4, 3 * np.pi / 4]])
WORKING_VELOCITIES = np.pi/180*np.array([[-1, 1], [-1, 1]])
# Screen Size
WIDTH, HEIGHT = 640, 480

# Screen Center
START = v(WIDTH // 2, HEIGHT // 2)


def _to_zero(vector) -> v:
    # make rigid transformation to move the vector to the center of the screen
    return vector + START


# Environment Constants
ALPHA = 1e-3
GAMMA = 1
SIGMA = 1e-2
NUM_EPISODES = 1e3
M = 5
N = 4
A = 3