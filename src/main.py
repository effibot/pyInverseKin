"""
Main file for the simulation of the robot arm.
We define the robot and simulate its motion in a 2D space using the
matplotlib animation module.
"""

import Robot
import Trajectory
import sys
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def main(arm):
    # Create a trajectory
    trj = Trajectory.Trajectory()
    # rectangle trajectory
    x, y = trj.generate_rectangle([-0.25, 0.25], [0.15, 0.15], 0.0)

    trajectory_str = []
    # Initial (Start) Position
    trajectory_str.append(
        {
            "interpolation": "joint",
            "start_p": [0.50, 0.0],
            "target_p": [x[0], y[0]],
            "step": 50,
            "cfg": 0,
        }
    )

    for i in range(len(x) - 1):
        trajectory_str.append(
            {
                "interpolation": "linear",
                "start_p": [x[i], y[i]],
                "target_p": [x[i + 1], y[i + 1]],
                "step": 25,
                "cfg": 0,
            }
        )

    animator = animation.FuncAnimation(
        arm.figure,
        arm.start_animation,
        init_func=arm.init_animation,
        frames=len(arm.trajectory[0]),
        interval=2,
        blit=True,
        repeat=False,
    )


if __name__ == "__main__":
    # run the simulation
    print("Starting simulation...")
    # create the robot
    working_range = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    arm = Robot.Robot("2 DoF Planar Robot", 0.4, 0.2, working_range)
    arm.display()
