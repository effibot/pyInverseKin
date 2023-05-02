import sys, pygame
from Robot2 import Robot
import matplotlib.pyplot as plt
import numpy as np
from pygame.math import Vector2 as v
import constants as c

# CONSTANTS
running = True
start = v(c.WIDTH // 2, c.HEIGHT // 2)
target = v(start.x + c.L1 / 3 * 2 + c.L2, start.y + 50)


def draw_axis(robot, ws, ax=None, **plt_kwargs):
    # get the figure and axes
    if ax is None:
        ax = plt.gca()
    # set general plot properties
    ax.set_title(robot.robot_name)
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-c.WIDTH * 0.75, c.WIDTH * 0.75)
    ax.set_ylim(-c.HEIGHT * 0.75, c.HEIGHT * 0.75)
    # draw axis
    ax.hlines(0, -c.WIDTH, c.WIDTH, color="k", linewidth=0.1)
    ax.vlines(0, -c.HEIGHT, c.HEIGHT, color="k", linewidth=0.1)
    # plot the workspace
    ax.scatter(ws[0], ws[1], s=0.1, c="g", alpha=0.5)
    # Make the plot
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    buffer = canvas.buffer_rgba()  # type: ignore
    surf = pygame.image.frombuffer(buffer, canvas.get_width_height(), "RGBA")
    plt.clf()
    return surf


def make_axis(ws, ax=None, **plt_kwargs):
    # define number of horizontal lines

    for i, j in zip(
        range(0, c.HEIGHT + 1, c.HEIGHT // ((c.HEIGHT) % 7)),
        range(0, c.WIDTH + 1, c.WIDTH // ((c.WIDTH) % 6)),
    ):
        # horizontal lines
        pygame.draw.line(screen, c.BLACK, (0, i), (c.WIDTH, i))
        # vertical lines
        pygame.draw.line(screen, c.BLACK, (j, 0), (j, c.WIDTH))
        ## add the principal axis
    # draw the horizontal axis
    # pygame.draw.line(screen, c.BLACK, (0, c.HEIGHT // 2), (c.WIDTH, c.HEIGHT // 2))
    # draw the vertical axis
    # pygame.draw.line(screen, c.BLACK, (c.WIDTH // 2, 0), (c.WIDTH // 2, c.HEIGHT))


def render(robot, ws):
    screen.fill(c.WHITE)
    # surf = draw_axis(robot, ws)
    # screen.blit(surf, (0, 0))
    # h = surf.get_c.HEIGHT()
    # w = surf.get_c.WIDTH()
    # draw the robot
    make_axis(screen)
    robot.render(screen)
    # # draw target
    pygame.draw.circle(screen, c.BLACK, (int(target[0]), int(target[1])), 7.5)
    pygame.draw.circle(screen, c.RED, (int(target[0]), int(target[1])), 5)
    pygame.display.flip()


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
    arm = Robot(c.NAME, c.L1, c.L2, wr, center)
    n = 100
    ws = arm.generate_ws_curves(n)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        render(arm, ws)
        clock.tick(60)
    pygame.quit()
