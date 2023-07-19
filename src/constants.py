
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
L1 = 0.3
L2 = 0.2

# Link Masses
M1 = 1
M2 = 0.7

# Gravity
g = -9.81

# Friction
f1 = 1
f2 = 1

# Robot Name
NAME = "2 DoF Planar Robot"

# Working Range (in radians)
WORKING_RANGE = np.array([[-np.pi, np.pi], [-5 * np.pi / 6, 5 * np.pi / 6]], dtype=np.float64)
WORKING_VELOCITIES =  (np.pi)*np.array([[-1, 1], [-1, 1]], dtype=np.float64)
U1 = 1e2
U2 = 1e2
# Screen Size
WIDTH, HEIGHT = 640, 480

# Screen Center
START = v(WIDTH // 2, HEIGHT // 2)

# Number of features
NUM_FEATURES = 2048

# Number of actions
NUM_ACTIONS = 9

# Number of Tiling
NUM_TILINGS = 8

# Target
TARGET = np.array([L1 / 3 * 2 + L2 / 4, L1 / 3 * 2 + L2 / 4])


def _to_zero(vector) -> v:
    # make rigid transformation to move the vector to the center of the screen
    return vector + START


# Environment Constants
ALPHA = 1/(10*NUM_TILINGS)
GAMMA = 1
SIGMA = 1e-2
EPS_START = 0.3
EPS_DECAY = 1e-2
NUM_EPISODES = int(2e3)
TIME_STEP = 1e-3  # time.Clock().tick(60) / 1e3
MAX_STEPS = int(3e3)
init_cond = np.asarray([-np.pi / 4, np.pi / 2, 0, 0], dtype=np.float64)
R_SCALE = 1.1
