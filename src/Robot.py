"""
Python module to represent a 2 DoF planar robot.
This implementation uses the closed-form inverse kinematics solution for faster computation.

The robot is drawn with a simple square representing the base (and the origin of the coordinate system) and two lines representing the links.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Robot(object):
    """Simple python class to represent a 2 DoF planar robot."""

    def __init__(
        self,
        name: str,
        l1: float,
        l2: float,
        working_range=[[-np.pi, np.pi], [-np.pi, np.pi]],
    ) -> None:
        """Initialize the robot with the given link lengths."""
        # robot name
        self.robot_name: str = name
        # Robot link lengths
        self.__l1: float = l1
        self.__l2: float = l2
        # Robot pose
        self.__pose: np.ndarray = np.array([0, 0])
        # Robot joint angles
        self.__Joint_angles: np.ndarray = np.array([0, 0])
        # Robot joint constraints
        self.working_range: np.ndarray = working_range
        # Robot Jacobian matrix - for inverse kinematics computation with the Jacobian pseudoinverse
        self.__J = np.zeros((2, 2))
        # Resulting trajectory
        self.trajectory = [[], [], []]
        

    def forward_kinematics(self, q1, q2):
        """Compute the forward kinematics of the robot.

        Args:
            q1: The angle of the first joint. Rad.
            q2: The angle of the second joint. Rad.

        Returns:
            A tuple containing the x and y coordinates of the end effector.
        """
        x = self.__l1 * np.cos(q1) + self.__l2 * np.cos(q1 + q2)
        y = self.__l1 * np.sin(q1) + self.__l2 * np.sin(q1 + q2)
        return np.asarray((x, y))

    def inverse_kinematics(self, x, y, elbow_up=True):
        """Compute the inverse kinematics of the robot.

        Args:
            x: The x coordinate of the end effector. Meter.
            y: The y coordinate of the end effector. Meter.

        Returns:
            A tuple containing the angles of the two joints.
        """
        # compute the angle of the second joint and find the two possible solutions
        c2 = (x**2 + y**2 - self.__l1**2 - self.__l2**2) / (
            2 * self.__l1 * self.__l2
        )
        # find the two possible solutions for the second joint
        if c2 > 1 or c2 < -1:
            # no feasible solution, fit the elbow to the nearest feasible position
            c2 = np.clip(c2, -1, 1)
            q2 = np.arctan2(np.sqrt(1 - c2**2), c2)
        else:
            # feasible solution, compute the two possible solutions for selected elbo
            if elbow_up:
                q2 = np.arctan2(np.sqrt(1 - c2**2), c2)
            else:
                q2 = np.arctan2(-np.sqrt(1 - c2**2), c2)
        # compute the angle of the first joint
        if (c2 == -1) | (self.__l1 == self.__l2):
            # if so, q1 is singular and we have infinite^1 solutions
            # in this case use the geometric solution
            q1 = np.clip(
                np.arctan2(y, x)
                - np.arctan2(
                    self.__l2 * np.sin(q2), self.__l1 + self.__l2 * np.cos(q2)
                ),
                -np.pi - 1e-6,  # we add a small offset to avoid numerical issues
                np.pi,
            )
        else:
            # use the algebraic solution - we skip the denominator since it is always positive
            c1_num = x * (self.__l1 + self.__l2 * np.cos(q2)) + y * self.__l2 * np.sin(
                q2
            )
            s1_num = y * (self.__l1 + self.__l2 * np.cos(q2)) - x * self.__l2 * np.sin(
                q2
            )
            q1 = np.arctan2(s1_num, c1_num)
        return np.asarray((q1, q2))

    def jacobian(self, q1, q2):
        """Compute the Jacobian matrix of the robot.

        Args:
            q1: The angle of the first joint.
            q2: The angle of the second joint.

        Returns:
            The Jacobian matrix of the robot.
        """
        J = np.zeros((2, 2))
        J[0, 0] = -self.__l1 * np.sin(q1) - self.__l2 * np.sin(q1 + q2)
        J[0, 1] = -self.__l2 * np.sin(q1 + q2)
        J[1, 0] = self.__l1 * np.cos(q1) + self.__l2 * np.cos(q1 + q2)
        J[1, 1] = self.__l2 * np.cos(q1 + q2)
        self.__J = J
        return J

    def __get_pose(self):
        """Getter for the pose attribute."""
        return self.__pose, self.__Joint_angles

    def __get_jacobian(self):
        """Getter for the Jacobian attribute."""
        return self.__J

    def __set_pose_ik(self, pose):
        """Setter for the pose attribute."""
        self.__Joint_angles = self.inverse_kinematics(pose[0], pose[1])
        self.__pose = pose

    def __set_pose_fwd(self, joint_angles):
        """Setter for the pose attribute."""
        self.__pose = self.forward_kinematics(joint_angles[0], joint_angles[1])
        self.__Joint_angles = joint_angles

    def __is_inside_ws(self, pose):
        """Check if the given pose is inside the workspace of the robot."""
        x, y = pose
        return ((x**2 + y**2) < (self.__l1 + self.__l2) ** 2) & (
            x**2 + y**2 > (self.__l1 - self.__l2) ** 2
        )

    def generate_ws_curves(self, n):
        """Generate the workspace curves of the robot brute-forcing the forward kinematics."""
        # define how many points to use to draw the curves. The higher the number, the smoother the curve
        n_points = n
        # define the range of angles to use
        t1 = np.linspace(self.working_range[0][0], self.working_range[0][1], n_points)[
            :-1
        ]
        t2 = np.linspace(self.working_range[1][0], self.working_range[1][1], n_points)[
            :-1
        ]
        # create the grid of points
        grid = np.meshgrid(t1, t2)
        # compute the forward kinematics for each point in the grid
        X, Y = self.forward_kinematics(grid[0], grid[1])
        # filter out the points that are outside the workspace
        X, Y = X[self.__is_inside_ws((X, Y))], Y[self.__is_inside_ws((X, Y))]
        # filter out duplicate of pairs of points
        X, Y = np.unique([X, Y], axis=1)
        # return the points
        return X, Y