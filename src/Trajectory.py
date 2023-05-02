"""Module for trajectory generation with B-splines and cubic splines."""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class Trajectory(object):
    def __init__(self):
        self.trajectory = [[], [], []]
        self.plt = plt
        pass

    def generate_rectangle(self, centroid, dimension, angle):
        """
        Description:
            A simple function to generate a path for a rectangle.
        Args:
            (1) centroid [Float Array]: Centroid of the Rectangle (x, y).
            (2) dimension [Float Array]: Dimensions (width, height).
            (3) angle [Float]: Angle (Degree) of the Rectangle.

        Returns:
            (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of path values.
        Examples:
            generate_rectangle([1.0, 1.0], [1.0, 1.0], 0.0)
        """

        # (loop of) Corner coordinates of the Rectangle: btm-left, btm-right, top-right, top-left, btm-left
        p = [
            [
                (-1) * dimension[0] / 2,
                (-1) * dimension[0] / 2,
                (+1) * dimension[0] / 2,
                (+1) * dimension[0] / 2,
                (-1) * dimension[0] / 2,
            ],
            [
                (-1) * dimension[1] / 2,
                (+1) * dimension[1] / 2,
                (+1) * dimension[1] / 2,
                (-1) * dimension[1] / 2,
                (-1) * dimension[1] / 2,
            ],
        ]

        x = []
        y = []

        for i in range(len(p[0])):
            # Calculation position of the Rectangle
            x.append(
                (
                    p[0][i] * np.cos(angle * (np.pi / 180))
                    - p[1][i] * np.sin(angle * (np.pi / 180))
                )
                + centroid[0]
            )
            y.append(
                (
                    p[0][i] * np.sin(angle * (np.pi / 180))
                    + p[1][i] * np.cos(angle * (np.pi / 180))
                )
                + centroid[1]
            )

        return [x, y]

    def smooth_trajectory(self, trajectory_str):
        """
        Description:
            A Bézier curve is a parametric curve used in computer graphics and related fields.
            The function shows several types of smoothing trajectories using Bézier curves (Quadratic, Cubic).

        Args:
            (1) trajectory_str [Structure Type Array]: Structure of the trajectory points.

        Returns:
            (1 - 2) parameter{1}, parameter{2} [Float Array]: Results of trajectory values (x, y).
            (3) parameter{3} [INT Array]: Results of trajectory values (Inverse Kinematics config).
        """

        try:
            assert len(trajectory_str) == 2 or len(trajectory_str) == 3

            if len(trajectory_str) == 2:
                cfg = []
                time = np.linspace(
                    0.0, 1.0, trajectory_str[0]["step"] + trajectory_str[1]["step"]
                )

                # Quadratic Bezier Curve
                # p(t) = ((1 - t)^2)*p_{0} + 2*t*(1 - t)*p_{1} + (t^2)*p_{2}, t ∈ [0, 1]
                x = (
                    ((1 - time) ** 2) * trajectory_str[0]["start_p"][0]
                    + 2 * (1 - time) * time * trajectory_str[0]["target_p"][0]
                    + (time**2) * trajectory_str[1]["target_p"][0]
                )
                y = (
                    ((1 - time) ** 2) * trajectory_str[0]["start_p"][1]
                    + 2 * (1 - time) * time * trajectory_str[0]["target_p"][1]
                    + (time**2) * trajectory_str[1]["target_p"][1]
                )

                cfg.append([trajectory_str[0]["cfg"]] * trajectory_str[0]["step"])
                cfg.append([trajectory_str[1]["cfg"]] * trajectory_str[1]["step"])
            else:
                cfg = []
                time = np.linspace(
                    0.0,
                    1.0,
                    trajectory_str[0]["step"]
                    + trajectory_str[1]["step"]
                    + trajectory_str[2]["step"],
                )

                # Cubic Bezier Curve
                # p(t) = ((1 - t)^3)*p_{0} + 3*t*((1 - t)^2)*p_{1} + (3*t^2)*(1 - t)*p_{2} + (t^3) * p_{3}, t ∈ [0, 1]
                x = (
                    ((1 - time) ** 3) * (trajectory_str[0]["start_p"][0])
                    + (3 * time * (1 - time) ** 2) * (trajectory_str[0]["target_p"][0])
                    + 3 * (time**2) * (1 - time) * trajectory_str[1]["target_p"][0]
                    + (time**3) * trajectory_str[2]["target_p"][0]
                )
                y = (
                    ((1 - time) ** 3) * (trajectory_str[0]["start_p"][1])
                    + (3 * time * (1 - time) ** 2) * (trajectory_str[0]["target_p"][1])
                    + 3 * (time**2) * (1 - time) * trajectory_str[1]["target_p"][1]
                    + (time**3) * trajectory_str[2]["target_p"][1]
                )

                cfg.append([trajectory_str[0]["cfg"]] * trajectory_str[0]["step"])
                cfg.append([trajectory_str[1]["cfg"]] * trajectory_str[1]["step"])
                cfg.append([trajectory_str[2]["cfg"]] * trajectory_str[2]["step"])

            self.plt.plot(x, y, "--", c=[0.1, 0.0, 0.7, 1.0], linewidth=3.0)

            return [x, y, np.concatenate(cfg)]

        except AssertionError:
            print("[INFO] Insufficient number of entry points.")
            print(
                "[INFO] The number of entry points must be 3 (Quadratic Curve) or 4 (Cubic Curve)."
            )

            return False
