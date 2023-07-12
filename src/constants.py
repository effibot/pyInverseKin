from re import M
from tkinter import E
import numpy as np
from pygame.math import Vector2 as v
from pygame import Surface, draw, transform, time

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

# Link Masses
M1 = 2
M2 = 1

# Gravity
g = -9.81

# Robot Name
NAME = "2 DoF Planar Robot"

# Working Range (in radians)
WORKING_RANGE = np.array([[-np.pi, np.pi], [-5 * np.pi / 6, 5 * np.pi / 6]], dtype=np.float64)
WORKING_VELOCITIES = np.pi * np.array([[-1, 1], [-1, 1]], dtype=np.float64)
# Screen Size
WIDTH, HEIGHT = 640, 480

# Screen Center
START = v(WIDTH // 2, HEIGHT // 2)

# Number of features
NUM_FEATURES = 1024

# Number of actions
NUM_ACTIONS = 9

# Number of Tiling
NUM_TILINGS = 8


def _to_zero(vector) -> v:
    # make rigid transformation to move the vector to the center of the screen
    return vector + START


# Environment Constants
ALPHA = 0.5
GAMMA = 0.5
SIGMA = 1e-2
EPSILON = 1e-3
NUM_EPISODES = int(1e5)
TIME_STEP = 1e-3  # time.Clock().tick(60) / 1e3
MAX_STEPS = int(2e3)
init_cond = [
    np.asarray(
        [
            -np.pi / 4,
            np.pi / 2,
        ]
    ),
    np.asarray([0, 0]),
    np.asarray([0, 0]),
]
