import sys, pygame
from Robot2 import Robot
import matplotlib.pyplot as plt
import numpy as np
from pygame.math import Vector2 as v
import constants as c
from pygame import font

# CONSTANTS
running = True
target = c._to_zero(v(c.L1 / 3 * 2 + c.L2, 50))


def make_axis(ws, ax=None, **plt_kwargs):
    # define number of horizontal lines
    x_range = range(0, c.HEIGHT + 1, c.HEIGHT // ((c.HEIGHT) % 7))
    y_range = range(0, c.WIDTH + 1, c.WIDTH // ((c.WIDTH) % 6))
    for i, j in zip(x_range, y_range):
        if i == c.HEIGHT // 2 or j == c.WIDTH // 2:
            color = c.BLACK
        else:
            color = c.GREY
        # horizontal lines
        pygame.draw.line(screen, color, (0, i), (c.WIDTH, i))
        # vertical lines
        pygame.draw.line(screen, color, (j, 0), (j, c.WIDTH))
    # Add X-axis positive indicator
    pygame.draw.polygon(
        screen,
        c.BLACK,
        [
            (c.WIDTH - 10, c.HEIGHT // 2 - 5),
            (c.WIDTH - 10, c.HEIGHT // 2 + 5),
            (c.WIDTH, c.HEIGHT // 2),
        ],
    )
    # Add Y-axis positive indicator
    pygame.draw.polygon(
        screen,
        c.BLACK,
        [
            (c.WIDTH // 2 - 5, 10),
            (c.WIDTH // 2 + 5, 10),
            (c.WIDTH // 2, 0),
        ],
    )
    # Add Axis Lables
    if pygame.font:
        font = pygame.font.Font(None, 20)
        # Add X label to the positive indicator
        text = font.render("X", True, c.BLACK)
        screen.blit(text, (c.WIDTH - 10, c.HEIGHT // 2 - 20))
        # Add Y label to the positive indicator
        text = font.render("Y", True, c.BLACK)
        screen.blit(text, (c.WIDTH // 2 + 10, 0))
    # Add corners
    pygame.draw.polygon(screen, c.GREY, [(c.WIDTH - 5, 0), (c.WIDTH, 0), (c.WIDTH, 5)])
    pygame.draw.polygon(
        screen,
        c.GREY,
        [(c.WIDTH - 5, c.HEIGHT), (c.WIDTH, c.HEIGHT), (c.WIDTH, c.HEIGHT - 5)],
    )
    pygame.draw.polygon(screen, c.GREY, [(0, 0), (5, 0), (0, 5)])
    pygame.draw.polygon(
        screen, c.GREY, [(0, c.HEIGHT), (5, c.HEIGHT), (0, c.HEIGHT - 5)]
    )


def render(robot, ws):
    # reset the background
    screen.fill(c.WHITE)
    # draw the axis
    make_axis(screen)
    # draw the robot
    robot.render(screen, True)
    # # draw target
    pygame.draw.circle(screen, c.BLACK, (int(target[0]), int(target[1])), 7.5)
    pygame.draw.circle(screen, c.RED, (int(target[0]), int(target[1])), 5)
    pygame.display.flip()


from Environment import Environ

if __name__ == "__main__":
    pygame.init()
    size = c.WIDTH, c.HEIGHT
    flags = pygame.DOUBLEBUF
    screen = pygame.display.set_mode(size, flags)
    surface = screen.get_rect()
    center = np.array(surface.center)
    clock = pygame.time.Clock()
    wr = c.WORKING_RANGE
    # arm = Robot("2 DoF Planar Robot", l1, l2, working_range)
    arm = Robot(c.NAME, c.L1, c.L2, working_range=wr, center=center)
    env = Environ(arm)
    n = 100
    ws = arm.generate_ws_curves(n)
    arm.set_target(target)
    env.get_info()
    print(env.get_indexes(arm.get_state()))
    # while running:
    #    for event in pygame.event.get():
    #        if event.type == pygame.QUIT:
    #            sys.exit()
    #        elif event.type == pygame.KEYDOWN:
    #            if event.key == pygame.K_ESCAPE:
    #                running = False
    #    render(arm, ws)
    # arm.set_joint_angles(arm.get_q() + [0.01, 0.01])
    env.SARSA()

    #    clock.tick(60)
    # pygame.quit()
